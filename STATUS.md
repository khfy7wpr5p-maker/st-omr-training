# ST-OMR Training Lab Status

Updated: 2026-09-14

This is the current stage-status source. `ARCHITECTURE.md` preserves historical detail; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- protected `main`: `79c2631682ebdb3b2be30c146a191e3ef8183ad0`
- latest merged package: PR #156 — TR-POLY-09B2 deterministic V2 metric/adaptor layer
- active package: TR-POLY-09B3 common VALIDATION aggregation/reporting
- current Poly2D trainer limit: maximum 2 smoke steps
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Current stage status

| Stage / package | Description | Status |
|---|---|---|
| 0–6 | Deterministic symbolic → rendered → validated synthetic dataset | ✅ Closed / preserved |
| 7-A/B/C | Baseline model/training evidence | ✅ Historical baseline |
| 7-D | Specialist architecture/evidence | ✅ Historical evidence |
| 8-0/1/2 | Real-data rights/intake/run contracts | ✅ Preserved |
| TR-POLY-02 | Evaluation taxonomy + benchmark identity | ✅ Closed / frozen |
| TR-POLY-03/04 | External-data registry + benchmark harness | ✅ Closed |
| TR-POLY-05/06 | V2 representation + tokenizer/roundtrip | ✅ Closed / frozen |
| TR-POLY-07/08/08A/08B | Registry + 2D model + smoke trainer + checkpoint | ✅ Research implementation |
| TR-POLY-08C | Stage 6 V1→V2 execution | ✅ Single-voice evidence only |
| TR-POLY-09A | Native explicit V2 dataset/materialization | ✅ Merged |
| TR-POLY-09B1 | Free-running greedy inference | ✅ Merged |
| TR-POLY-09B2 | Deterministic V2 metric/adaptor layer | ✅ Merged / CI green |
| TR-POLY-09B3 | Common VALIDATION aggregation/report | ✅ Implemented in active package / merge gate pending |
| Smoke VALIDATION execution | End-to-end measurement sanity run | 🔄 After B3 merge |
| Quality-training regime | Multi-step/epoch TRAIN-only candidate training | 🔒 Not implemented yet |
| Quality VALIDATION benchmark | Measure a genuinely trained candidate | 🔒 After quality training |
| Missing metric admission | 5 frozen metrics remain unsupported | 🔒 Separate packages required |
| P09C | Evidence-driven refinement | 🔒 After quality VALIDATION evidence |
| Stage 9 | Final sealed TEST decision | 🔒 TEST sealed |
| Stage 10 | ScoreMosaic shadow/integration | 🔒 Not started |

## B1–B2 completed

B1 closed the teacher-forcing gap and produces strict free-running V2 prediction evidence from BOS alone. B2 maps each VALIDATION prediction/reference pair to the frozen TR-POLY-02 metric vocabulary.

B2 provides 11 numeric metrics:

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

Five metrics remain explicitly unsupported:

```text
musicxml_validity
tedn
notehead_stem_f1
beam_relation_f1
tie_relation_f1
```

No proxy value is permitted under those frozen metric IDs.

## B3 implemented behavior

TR-POLY-09B3 aggregates only checkpoint-bound B2 VALIDATION reports that share exactly one benchmark identity and one candidate identity.

The aggregation policy is `sample-macro-mean-v1`.

B3 produces:

- one overall VALIDATION slice;
- separate `1_voice`, `2_voice`, `3_voice`, `4_plus_voice` slices when present;
- explicit `missing_voice_strata` when coverage is incomplete;
- separate observed robustness-bucket slices;
- parse-success counts/rates;
- mean/min/max for each available metric;
- explicit unsupported metric coverage/reasons;
- deterministic report fingerprint bound to exact sample-report fingerprints.

Invalid/abstain outputs remain in the denominator because B2 keeps them as scored sample reports.

## Training-readiness boundary

The current `Poly2DTrainingConfig` is deliberately a smoke harness and caps `smoke_steps` at 2. Existing checkpoint generation therefore proves deterministic training/checkpoint plumbing, not a converged or quality-trained OMR model.

Consequences:

- B3 can be used immediately for an end-to-end **sanity baseline** of the smoke checkpoint;
- those numbers must not be presented as the expected quality ceiling of the architecture;
- a genuine model-quality benchmark requires a separately versioned TRAIN-only training regime with a meaningful step/epoch budget, deterministic checkpoint selection and read-only VALIDATION selection evidence;
- TEST remains sealed while that training regime is developed and tuned.

## B3 fail-closed gates

B3 rejects:

- TRAIN or TEST evidence;
- unbound checkpoint evidence;
- mixed benchmark identities;
- mixed candidate/checkpoint identities;
- duplicate sample IDs;
- unexpected strata/buckets;
- mixed metric availability inside one aggregate.

`common_comparison_ready` remains false while any required metric is unsupported or any required voice stratum is missing. Candidate comparison validates comparability only; it does not rank or select a winner.

## What a green B3 merge means

A green merge means the repository has deterministic infrastructure to summarize VALIDATION evidence correctly. It does **not** mean a quality-trained checkpoint exists or that model-quality numbers already exist.

## Next gate

1. merge B3 only after exact-head CI is green;
2. optionally run the current ≤2-step checkpoint as a B1→B2→B3 sanity baseline;
3. implement a versioned quality-training regime on TRAIN only;
4. train and freeze the first quality candidate checkpoint;
5. run that checkpoint across the exact admitted VALIDATION manifest/build;
6. aggregate B1→B2→B3 evidence and inspect 1/2/3/4+ voice and robustness results;
7. choose P09C refinement from measured quality failures only;
8. keep TEST sealed throughout.
