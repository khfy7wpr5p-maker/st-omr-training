# ST-OMR Training Lab Status

Updated: 2026-09-14

This file is the current stage-status source for the repository. `ARCHITECTURE.md` keeps the long-form historical architecture; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- protected-main baseline for this package: `2bb0ac23e11fb953bb3ada4aaef6369073b7e1f1`
- latest previously merged technical package: PR #153 — TR-POLY-09A native Polyphonic V2 dataset materialization
- PR #154 architecture refresh: merged
- active package: TR-POLY-09B1 bounded free-running greedy inference
- current top-level lane: Polyphonic V2 candidate measurement readiness
- next implementation after TR-POLY-09B1: TR-POLY-09B2 deterministic metric/adaptor support
- next comparative execution: TR-POLY-09B3 common VALIDATION benchmark
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Current stage status

| Stage / package | Description | Status |
|---|---|---|
| 0–6 | Deterministic symbolic → rendered → validated synthetic dataset pipeline | ✅ Closed / preserved |
| 7-A/B/C | Baseline tokenizer/model/trainer + bounded baseline evidence | ✅ Closed / historical baseline |
| 7-D specialist lane | Staff/structure/local specialist architecture and evidence | ✅ Historical evidence / not current top-level lane |
| 8-0/1/2 | Real-data rights, intake and paired-run contracts | ✅ Closed / preserved |
| 8-3 real-data execution | Real pilot execution path | ⏸ Not current priority |
| TR-POLY-02 | Polyphonic evaluation taxonomy + benchmark identity | ✅ Closed |
| TR-POLY-03 | External dataset/license registry | ✅ Closed |
| TR-POLY-04 | Deterministic external benchmark harness | ✅ Closed |
| TR-POLY-05 | Polyphonic Representation V2 | ✅ Closed / frozen |
| TR-POLY-06 | V2 parser/tokenizer/lossless roundtrip | ✅ Closed |
| TR-POLY-07 | Research model registry | ✅ Closed |
| TR-POLY-08 | Tiny 2D Transformer architecture | ✅ Implemented / research |
| TR-POLY-08A | Bounded training + provenance | ✅ Implemented |
| TR-POLY-08B | Exact checkpoint persistence/reload | ✅ Implemented |
| TR-POLY-08C | Exact Stage 6 V1→V2 artifact execution | ✅ Closed / single-voice evidence only |
| TR-POLY-09A | Native explicit Polyphonic V2 dataset/materialization | ✅ Merged on protected main |
| TR-POLY-09B1 | Free-running greedy decode + strict V2 semantic validation | ✅ Implemented in active package / merge gate pending |
| TR-POLY-09B2 | Deterministic common metric implementations/adapters | 🔄 Next after B1 merge |
| TR-POLY-09B3 | Common comparable VALIDATION benchmark | 🔒 After B2 |
| Polyphonic refinement | Evidence-driven TRAIN/VALIDATION refinement | 🔒 After benchmark |
| Stage 9 sealed TEST | Final held-out candidate decision | 🔒 TEST sealed |
| Stage 10 | Separate ScoreMosaic shadow/integration gate | 🔒 Not started |

## Latest architecture chain

```text
TR-POLY-02 evaluation contract
        ↓
TR-POLY-03 rights/license registry
        ↓
TR-POLY-04 external benchmark harness
        ↓
TR-POLY-05 structured Polyphonic V2
        ↓
TR-POLY-06 parser/tokenizer/roundtrip
        ↓
TR-POLY-07 model registry
        ↓
TR-POLY-08 2D Transformer
        ↓
TR-POLY-08A bounded training/provenance
        ↓
TR-POLY-08B checkpoint/reload
        ↓
TR-POLY-08C exact Stage 6 V1→V2 execution
        ↓
TR-POLY-09A native Polyphonic V2 materialization
        ↓
TR-POLY-09B1 free-running greedy inference + strict semantic validation
```

## TR-POLY-09B1 scope

The package closes the teacher-forcing gap required before end-to-end OMR comparison.

```text
one admitted image
        ↓
2D visual memory computed once
        ↓
BOS-only prefix
        ↓
deterministic greedy token generation
        ↓
EOS / invalid-control / max-step termination
        ↓
strict V2 reconstruction
        ↓
canonical prediction OR explicit invalid/abstain evidence
```

The model does not receive a gold target prefix. Generated PAD or a second BOS fails closed rather than being silently masked. Decode-limit exhaustion does not fabricate EOS. EOS is accepted only if strict V2 detokenization/parser validation succeeds.

The inference path preserves model state and binds exact model/tokenizer/runtime identity. When called through the verified checkpoint wrapper it additionally binds checkpoint, metadata, receipt, dataset-manifest, preprocessing, trainer, provenance, registry and repository identities.

This is measurement infrastructure, not quality evidence. No accuracy or benchmark claim is made by implementing inference.

## Next required capability

TR-POLY-09B2 should connect free-running predictions and references to the frozen TR-POLY-02 evaluation contract.

Priority order:

1. serialization/parse result mapping;
2. token sequence metrics;
3. pitch/duration/onset/voice/staff semantic metrics;
4. relation metrics where the representation supplies both prediction and reference relations;
5. structural metric only after the exact algorithm/license surface is versioned and admitted.

Unavailable metrics must remain explicitly unsupported; they must not be guessed or synthesized.

After B2 is green, TR-POLY-09B3 may run the first common VALIDATION benchmark with separate 1/2/3/4+ voice strata.

## Safety boundaries

- No automatic TEST access before the final Stage 9 gate.
- TRAIN is the only split allowed to update parameters.
- VALIDATION may guide bounded development but must not mutate model state during evaluation.
- External data must pass rights/license/install-pin admission.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Large datasets/checkpoints stay outside ordinary Git content.
- Candidate artifacts remain exact hash/provenance bound.
- Deterministic musical validators retain veto authority.
- Historical specialist evidence is preserved and must not be silently rewritten as current benchmark evidence.

## Next gate

Merge TR-POLY-09B1 only after exact-head CI is green. Then implement TR-POLY-09B2 metric/adaptor support and proceed to the TR-POLY-09B3 common VALIDATION benchmark. TEST remains sealed.
