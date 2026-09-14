# ST-OMR Training Lab Status

Updated: 2026-09-14

This is the current stage-status source. `ARCHITECTURE.md` preserves historical detail; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- protected `main`: `c1d6f3b9e1419839def00e64a41625d258ef2b47`
- latest merged package: PR #157 — TR-POLY-09B3 common VALIDATION aggregation
- active package: TR-POLY-09B4 deterministic quality-training regime
- frozen smoke trainer/checkpoint remain unchanged
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Current stage status

| Stage / package | Description | Status |
|---|---|---|
| 0–6 | Deterministic symbolic → rendered → validated synthetic dataset | ✅ Closed / preserved |
| 7-A/B/C | Baseline model/training evidence | ✅ Historical baseline |
| 7-D | Specialist architecture/evidence | ✅ Historical evidence |
| TR-POLY-02 | Evaluation taxonomy + benchmark identity | ✅ Frozen |
| TR-POLY-03/04 | External-data registry + benchmark harness | ✅ Closed |
| TR-POLY-05/06 | V2 representation + tokenizer/roundtrip | ✅ Frozen |
| TR-POLY-07/08 | Registry + 2D Transformer | ✅ Research implementation |
| TR-POLY-08A | ≤2-step deterministic smoke trainer | ✅ Frozen / preserved |
| TR-POLY-08B | ≤2-step exact research checkpoint | ✅ Frozen / preserved |
| TR-POLY-08C | Stage 6 V1→V2 execution | ✅ Single-voice evidence only |
| TR-POLY-09A | Native explicit V2 dataset/materialization | ✅ Merged |
| TR-POLY-09B1 | Free-running greedy inference | ✅ Merged |
| TR-POLY-09B2 | Deterministic V2 metric/adaptor layer | ✅ Merged |
| TR-POLY-09B3 | VALIDATION aggregation by voice/robustness | ✅ Merged / CI green |
| TR-POLY-09B4 | Multi-epoch TRAIN-only quality-training regime | ✅ Implemented in active package / merge gate pending |
| Quality checkpoint | Persist selected B4 candidate separately from smoke schema | 🔒 Next |
| Quality VALIDATION run | Exact quality checkpoint → B1 → B2 → B3 | 🔒 After quality checkpoint |
| Missing metric admission | Five frozen metrics remain unsupported | 🔒 Separate work |
| P09C | Evidence-driven refinement | 🔒 After quality VALIDATION evidence |
| Stage 9 | One-shot sealed TEST decision | 🔒 TEST sealed |
| Stage 10 | ScoreMosaic shadow/integration | 🔒 Not started |

## Measurement infrastructure now closed

The repository now contains the complete measurement control plane:

```text
B1 free-running inference
        ↓
B2 per-sample deterministic metrics
        ↓
B3 candidate-level VALIDATION aggregation
        ├─ overall
        ├─ 1 / 2 / 3 / 4+ voice
        └─ robustness buckets
```

B3 keeps invalid/abstain outputs in the population and refuses mixed benchmark/candidate identities.

## Why B4 is required

The historical Poly2D trainer is intentionally capped at two optimizer steps, and the associated checkpoint schema also rejects more than two steps. That path proves plumbing, not model quality.

B4 therefore creates a **separate** multi-epoch training regime instead of weakening the frozen smoke contract.

## B4 implemented behavior

B4 reuses the hardened TR-POLY-08A one-step primitive but adds a versioned multi-epoch control plane.

Frozen v1 policies:

- TRAIN-only parameter updates;
- caller-supplied immutable batch tuple order;
- no shuffle, scheduler or hidden sampling;
- full read-only VALIDATION after every epoch;
- arithmetic mean VALIDATION loss;
- fixed epoch budget;
- minimum mean VALIDATION-loss candidate selection;
- earliest epoch wins exact validation-loss ties;
- selected model state reloaded and hash-verified;
- TEST prohibited;
- production authority prohibited.

Safety ceilings:

- max 128 epochs;
- max 100,000 optimizer steps;
- max 100,000 supplied batches per split;
- all existing per-batch image/token limits remain active.

The exact training plan is fingerprinted, including ordered TRAIN/VALIDATION sample groups. Reordering batches changes the run identity.

## B4 evidence

Every epoch records:

- epoch number;
- cumulative optimizer steps;
- mean TRAIN loss;
- mean read-only VALIDATION loss;
- exact model-state SHA-256.

The run result additionally binds model profile, quality-training recipe, one-step primitive profile, dataset identity, tokenizer, repository SHA, runtime and quality provenance.

## Current metric surface

Numeric B2/B3 metrics:

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

Explicitly unsupported:

```text
musicxml_validity
tedn
notehead_stem_f1
beam_relation_f1
tie_relation_f1
```

No unsupported metric receives a proxy number.

## Next gate

1. merge B4 only after exact-head CI is green;
2. create a separate quality-checkpoint artifact contract; do not relax TR-POLY-08B;
3. train/freeze the first native-V2 quality candidate on TRAIN;
4. run that exact checkpoint over admitted VALIDATION artifacts;
5. produce B1→B2→B3 evidence by voice and robustness strata;
6. choose P09C refinement from measured failures only;
7. keep TEST sealed throughout.
