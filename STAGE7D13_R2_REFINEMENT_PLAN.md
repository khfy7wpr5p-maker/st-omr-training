# Stage 7-D13-R2 — Specialist refinement and recovery plan

> Historical/frozen recovery plan. This document preserves the architecture decision that followed the failed D13-R1 Rest run. It is no longer the current Rest status source. The current Rest R4 shadow-PASS state and runtime boundary are recorded in `ARCHITECTURE_CURRENT.md`.

Status: **historical architecture plan; R1 failure preserved, later Rest R4 recovery supersedes the active-plan status**.

## 2026-08-23 execution overlay

The plan below remains useful as causal history, but its original "next gate = Rest Root-Cause Audit" status is complete/superseded for Rest.

Current Rest shadow outcome:

```text
Half specialist      PASS
Quarter specialist   PASS
Eighth specialist    PASS
Deterministic class arbitration  PASS
Resolver connection  CLOSED
sealed TEST           CLOSED
production promotion CLOSED
```

The old R1 zero-F1 Rest result remains immutable baseline evidence and must not be deleted or rewritten. The next Rest-quality work is real-image/shadow E2E validation and evidence-selected refinement if needed, not a blind replay of the failed R1 training profile.

## Why R2 exists

The first D13 external specialist run is retained as an experimental baseline rather than accepted as a completed stage. The available evidence is sufficient to reject a simple "rerun the same training" response:

- NoteHead learned strongly and reached high validation F1, but minimum-validation-loss checkpoint selection did not always select the epoch with the strongest F1 metrics.
- Rest remained unusable in the observed run, with zero center/bbox/macro F1 at the end of the ten-epoch profile.
- Accidental was learning and improving before the run was interrupted, but macro-class performance still lagged the localization metrics.
- Colab/runtime restart recovery exposed avoidable operational latency because the full immutable derivative preflight was repeated before resume.

D13-R2 therefore treats D13-R1 artifacts, logs and checkpoints as immutable comparison evidence. R2 must not overwrite or silently relabel them as accepted production-quality results.

## R2 decision principles

1. **Do not rerun a failed architecture unchanged.** Root-cause evidence comes before another full 10-epoch Rest run.
2. **Keep successful learned structure.** NoteHead is not redesigned without evidence that its visual architecture is the limiting factor.
3. **Ask smaller visual questions.** Detection/localization and fine-grained symbol classification may be separated when a single objective couples them poorly.
4. **Preserve deterministic authority.** Learned observations remain subordinate to deterministic spatial/musical composition and validation.
5. **No sealed TEST access.** R2 design, diagnostics, threshold choices and checkpoint rules are development-only.
6. **Freeze selection rules before authoritative validation.** Scheduler policy, checkpoint ranking and diagnostic exit rules must be fixed before the result is known.
7. **Keep R1 evidence immutable.** R2 receives a new run identity, profile fingerprint, checkpoint lineage and verifier surface.

## Approved R2 sequence

```text
D13-R1 evidence freeze
        ↓
Rest root-cause audit
        ↓
Rest label/scale/decoder diagnostics
        ↓
Rest task decomposition decision
        ↓
short TRAIN-only diagnostic gate
        ↓
Rest R2 bounded training
        ↓
Accidental R2 refinement
        ↓
NoteHead optimizer/checkpoint refinement
        ↓
independent persisted-run verification
        ↓
product-oriented validation preparation
```

## R2-0 — Freeze the D13-R1 baseline

Persist and reference, without rewriting:

- repository SHA used by the R1 run;
- derivative build/manifest/artifact-binding identities;
- specialist epoch histories available from Drive logs/checkpoints;
- NoteHead selected checkpoint and per-epoch validation metrics;
- Rest failed-run evidence;
- Accidental partial-run evidence;
- runtime/preflight/resume logs.

The baseline exists to answer one question later: **did R2 actually improve the failing behavior under a controlled comparison?**

## R2-1 — Rest root-cause audit before optimizer work

No new Rest full training is authorized until the following failure modes are separated:

### Ground-truth audit

Verify representative and problem-focused Rest samples for:

- correct `half|quarter|eighth` class;
- correct bbox and center;
- target remains inside transformed measure bounds;
- no systematic class/geometry offset after letterbox transform;
- expected positive/negative measure distribution.

### Representation audit

Measure the Rest target after the existing 512×128 measure transform:

- pixel width/height distribution;
- staff-spacing-relative size;
- fraction of targets that become extremely small;
- distance to neighboring staff/note symbols;
- whether the global measure representation removes discriminative detail.

### Decoder audit

For the failed R1 Rest checkpoint inspect persisted predictions, or reproduce them without parameter updates, to distinguish:

- heatmap scores below the fixed 0.25 threshold;
- correct-near centers outside the 4 px matching boundary;
- localization that is plausible but class is wrong;
- class-correct centers with poor bbox IoU;
- no meaningful activation at all.

R2 must not tune the production validation threshold after seeing validation outcomes. Diagnostic threshold/distribution inspection is for failure attribution only; any changed decoder policy requires a separately frozen development rule.

## R2-2 — Rest architecture decision

The primary R2 candidate is a two-step observation pipeline:

```text
measure / local high-resolution ROI
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

The exact split is admitted only if R2-1 shows that joint localization + type classification is the limiting factor. If R2-1 instead finds bad labels or destructive scaling, fix that root cause first rather than adding model complexity.

## R2-3 — Specialist-specific visual scale

R2 no longer assumes that one 512×128 complete-measure representation is optimal for every symbol family.

Candidate policy:

```text
NoteHead    → keep current measure detector unless evidence says otherwise
Rest        → evaluate higher-resolution local ROI / tile representation
Accidental  → compare current detector against local ROI or detector→classifier split
```

Any new ROI must be deterministic, preserve complete target content, carry a reproducible transform fingerprint and retain TRAIN/VALIDATION family isolation.

## R2-4 — Short diagnostic gate before full training

To avoid spending hours on another clearly broken Rest run, introduce a bounded diagnostic stage that does **not** use sealed TEST and does not use the authoritative validation set for architecture-by-iteration.

Preferred design:

- derive a deterministic TRAIN-only diagnostic/dev partition by symbolic family;
- run a small fixed number of batches/steps;
- require evidence of non-zero, improving localization/classification behavior;
- fail fast if the model remains degenerate;
- only a passing candidate may enter the full R2 training profile.

The diagnostic threshold and maximum budget must be frozen before execution.

## R2-5 — Learning-rate scheduler experiment

Scheduler behavior becomes an explicit training-system research item rather than an ad-hoc addition.

Controlled candidates:

- A: fixed learning rate baseline;
- B: validation-driven scheduler such as `ReduceLROnPlateau`;
- C: predetermined decay such as cosine annealing.

Acceptance is based on stability and task metrics, not merely a lower scalar loss.

If a scheduler is admitted, resumable state must include:

```text
model state
optimizer state
scheduler state
completed epoch
optimizer-step count
current learning rate
learning-rate history
best-checkpoint selection state
```

The persisted verifier must independently check scheduler state and resume identity.

## R2-6 — Checkpoint-selection refinement

R1 showed that minimum validation loss and strongest F1 may identify different epochs. R2 therefore evaluates a checkpoint policy that is closer to the product task while remaining frozen before the run.

Candidate ordering:

1. candidate must pass all specialist technical gates;
2. prefer higher macro class F1;
3. use center/bbox F1 as secondary quality evidence;
4. use validation loss as a tie-break/regularization signal rather than the sole product proxy.

The exact deterministic ranking function must be specified and tested before optimizer execution. It must not be changed after viewing R2 validation results.

## R2-7 — Accidental refinement

Accidental is not automatically split because R1 showed meaningful learning.

First comparison:

```text
current detector
+ frozen scheduler candidate
+ refined checkpoint selection
+ per-class metrics
```

If one class, especially the rare class, remains materially weaker, evaluate:

```text
AccidentalPresence/Localization
        ↓
normalized crop
        ↓
sharp | flat | natural classifier
        ↓
deterministic note-association validator
```

Recognition of the accidental glyph remains distinct from association of that glyph to a note.

## R2-8 — NoteHead policy

NoteHead is **KEEP at the visual-architecture level** unless new evidence contradicts the R1 result.

R2 work for NoteHead is limited initially to:

- scheduler A/B evaluation;
- refined checkpoint selection;
- per-class stability review for `open|filled`;
- threshold-cliff diagnostics around score, center-distance and IoU boundaries.

A successful NoteHead detector must not be redesigned merely for architectural symmetry with Rest/Accidental.

## R2-9 — Safe fast resume

R1 restart handling was safe but expensive. R2 should support a fail-closed fast-resume path for immutable derivatives.

A fast resume may skip the full record-by-record preflight only when a persisted preflight receipt is cryptographically bound to all of:

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

If any identity differs, the full preflight is mandatory. Fast resume must never weaken TEST sealing, family isolation, checkpoint integrity or exact optimizer-step accounting.

## R2-10 — Evaluation that maps to ScoreMosaic quality

Technical F1 remains necessary but is not the final product question. R2 evidence should prepare later evaluation of:

- symbol detection correctness;
- symbol class correctness;
- pitch correctness after deterministic staff-position resolution;
- duration correctness after deterministic rhythm composition;
- accidental association correctness;
- exact note-event correctness;
- exact measure correctness;
- critical wrong decisions per 10,000 symbols;
- abstain/UNKNOWN behavior for unsafe confidence regions.

A specialist F1 must not be described as end-to-end PDF-reading accuracy.

## R2 acceptance boundary

D13-R2 is not complete merely because a model trains. Completion requires:

- Rest root cause documented;
- selected Rest architecture justified by evidence;
- short diagnostic gate passed before full training;
- all specialist checkpoint-selection rules frozen before authoritative validation;
- every required specialist technical gate passed;
- per-class metrics persisted;
- exact optimizer/scheduler/resume state verified;
- TEST remained sealed;
- independent persisted-run verifier passed;
- R1→R2 comparative decision table produced.

## Explicit non-goals

D13-R2 does not authorize:

- opening sealed TEST;
- automatic learning from ScoreMosaic uploads or teacher corrections;
- RhythmSet/PitchSet/ChordSet training;
- Meter V2 implementation;
- ScoreMosaic runtime integration;
- deleting D13-R1 evidence;
- treating a synthetic-validation F1 value as production PDF accuracy.

## Historical next implementation gate

The original first implementation package after this architecture document was **Rest Root-Cause Audit**. That gate is preserved here as history. Current Rest work must follow `ARCHITECTURE_CURRENT.md` and may not infer new optimizer or production authority from this historical plan.
