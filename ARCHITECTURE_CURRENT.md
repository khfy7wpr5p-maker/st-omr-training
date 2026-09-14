# ST-OMR Training — Current Architecture Overlay

Updated: 2026-09-14

This file records the active architecture lane. `ARCHITECTURE.md` remains the long-form historical record; `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` contains the detailed roadmap.

## Current baseline

- protected `main`: `c1d6f3b9e1419839def00e64a41625d258ef2b47`
- latest merged package: PR #157 — TR-POLY-09B3 common VALIDATION aggregation
- active package: TR-POLY-09B4 deterministic quality-training regime
- frozen smoke trainer/checkpoint: preserved unchanged
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Active pipeline

```text
Polyphonic Representation V2                         ✅ FROZEN
        ↓
V2 parser / tokenizer / lossless roundtrip          ✅
        ↓
Tiny 2D Transformer                                 ✅ RESEARCH
        ↓
≤2-step smoke trainer + research checkpoint         ✅ FROZEN INFRASTRUCTURE
        ↓
Native explicit Polyphonic V2 TRAIN/VALIDATION      ✅ TR-POLY-09A
        ↓
BOS-only free-running greedy inference               ✅ TR-POLY-09B1
        ↓
Deterministic per-sample metric/adaptor surface      ✅ TR-POLY-09B2
        ↓
VALIDATION aggregation by voice/robustness           ✅ TR-POLY-09B3
        ↓
Multi-epoch TRAIN-only quality-training regime       ✅ IMPLEMENTED — TR-POLY-09B4
        ↓
Separate quality-checkpoint artifact                 🔄 NEXT
        ↓
First quality-trained native-V2 candidate            🔒
        ↓
Quality VALIDATION: checkpoint → B1 → B2 → B3       🔒
        ↓
Evidence-driven P09C refinement                      🔒
        ↓
Missing metric admission                             🔒 5 surfaces
        ↓
P09D candidate/evaluation freeze                     🔒
        ↓
Stage 9 one-shot sealed TEST                         🔒
        ↓
Stage 10 ScoreMosaic shadow/integration              🔒
```

## Measurement path is now structurally complete

B1, B2 and B3 close the end-to-end evaluation control plane:

```text
image
  ↓
free-running V2 prediction
  ↓
strict parse or explicit invalid/abstain
  ↓
per-sample metrics
  ↓
overall + 1/2/3/4+ voice + robustness aggregation
```

Invalid predictions remain in denominators. Mixed benchmark identities, mixed candidate identities, duplicate samples and TEST evidence fail closed.

## The quality-training gap

The earlier Poly2D trainer and checkpoint were intentionally designed as bounded smoke evidence. Both are limited to at most two optimizer steps.

Therefore they cannot answer the question “How good can this architecture become after training?”

TR-POLY-09B4 closes the training-control-plane half of that gap without modifying the frozen smoke contracts.

## B4 quality-training architecture

```text
exact TRAIN batch tuple
        +
exact VALIDATION batch tuple
        +
quality config/provenance
        ↓
frozen-seed 2D Transformer initialization
        ↓
for each epoch:
    TRAIN batches in exact tuple order
    → gradient update through hardened 08A step primitive
    → full read-only VALIDATION
    → epoch loss + model-state evidence
        ↓
fixed epoch budget completes
        ↓
select lowest mean VALIDATION loss
        ↓
earliest epoch wins exact tie
        ↓
reload selected state
        ↓
verify exact selected-state SHA-256
```

### v1 policies

- no shuffle;
- no sampling;
- no learning-rate scheduler;
- no hidden early stopping;
- validation every epoch;
- full supplied VALIDATION set each epoch;
- selection by mean VALIDATION loss only;
- TRAIN/VALIDATION sample duplication rejected;
- one exact dataset-manifest identity across all batches;
- TEST unavailable;
- production authority false.

The ordered batch plan itself is SHA-256 fingerprinted, so changing batch order changes the candidate recipe identity.

## Resource bounds

B4 permits a real training regime while remaining bounded:

- maximum 128 epochs;
- maximum 100,000 optimizer steps;
- maximum 100,000 supplied batches per split;
- existing image geometry, token length, batch-size and finite-value limits remain inherited from TR-POLY-08A.

The exact planned step count is checked before training begins.

## Why checkpoint persistence remains separate

The TR-POLY-08B checkpoint schema explicitly validates `optimizer_steps <= 2`. Relaxing it in place would silently rewrite historical artifact semantics and could invalidate reproducibility.

Therefore the next package must introduce a distinct quality-checkpoint schema. It should persist the selected B4 state and exact B4 recipe/provenance while leaving the smoke artifact format untouched.

## Quality claims remain closed

A B4 green merge means multi-epoch deterministic training is implemented. It does not mean:

- a real native-V2 corpus has been trained;
- a quality checkpoint exists;
- the default recipe is optimal;
- any OMR accuracy has been measured;
- TEST has been opened;
- the model is ready for ScoreMosaic.

Quality evidence begins only after a persisted exact B4 candidate is run through B1→B2→B3 on VALIDATION.

## Current metric coverage

Available numeric metrics:

```text
parse_success
ter
normalized_edit_distance
exact_sequence_accuracy
pitch_accuracy
duration_accuracy
onset_accuracy
voice_accuracy
staff_accuracy
accidental_note_f1
note_staff_f1
```

Unsupported frozen metrics:

```text
musicxml_validity
tedn
notehead_stem_f1
beam_relation_f1
tie_relation_f1
```

Unsupported metrics remain nonnumeric until separately admitted.

## Required order from here

```text
B4 exact-head CI + merge
        ↓
quality-checkpoint schema + verified reload
        ↓
train/freeze first native-V2 quality candidate
        ↓
quality VALIDATION B1→B2→B3
        ↓
inspect measured failure strata
        ↓
P09C targeted refinement
        ↓
complete missing metric surface
        ↓
P09D freeze
        ↓
Stage 9 sealed TEST once
        ↓
Stage 10 shadow integration
```

## Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN alone changes model parameters.
- VALIDATION is read-only and used for candidate selection/evaluation only.
- Smoke evidence cannot be relabeled as quality evidence.
- Historical smoke schemas remain immutable.
- Invalid/abstain predictions remain visible in benchmark denominators.
- Unsupported metrics never receive proxy values.
- External data still requires rights/license/install-pin admission.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- No training or benchmark package grants production authority.

## Next gate

Merge TR-POLY-09B4 only after exact-head CI is green. Then add a separate quality-checkpoint artifact contract and verified B1 inference wrapper. Only after that should the first genuine native-V2 quality training/VALIDATION experiment be executed.
