# ST-OMR Training Lab Status

Updated: 2026-09-14

This file is the current stage-status source for the repository. `ARCHITECTURE.md` keeps the long-form historical architecture; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- protected branch: `main`
- reviewed main head: `6a6bf1e1faf0149ebd097a310536076b77feac65`
- latest merged package: PR #153 — TR-POLY-09A native Polyphonic V2 dataset materialization
- current top-level lane: Polyphonic V2 candidate measurement readiness
- next comparative gate: TR-POLY-09B common benchmark
- immediate missing capability before valid comparison: free-running Polyphonic V2 inference / semantic decoding
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
| Polyphonic V2 inference | Free-running greedy decode + semantic validation | 🔄 Next required capability |
| TR-POLY-09B | Common comparable VALIDATION benchmark | 🔒 Next gate |
| Polyphonic refinement | Evidence-driven TRAIN/VALIDATION refinement | 🔒 After benchmark |
| Stage 9 sealed TEST | Final held-out candidate decision | 🔒 TEST sealed |
| Stage 10 | Separate ScoreMosaic shadow/integration gate | 🔒 Not started |

## Latest completed architecture chain

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
```

## TR-POLY-09A accepted scope

The merged native V2 path admits explicit canonical Polyphonic V2 TRAIN/VALIDATION targets and grayscale PNG artifacts through hash-bound manifests/build identities into the existing 2D training/checkpoint chain.

The contract includes explicit polyphonic coverage gates for voice 2 and corpus-level voice 3 / voice 4+ cases, structural chord-vs-independent-voice distinction, exact onset/duration/staff/voice preservation, lossless tokenizer roundtrip, family leakage protection, semantic truncation rejection and sealed-TEST non-access.

It does not claim recognition accuracy, benchmark superiority, production readiness or ScoreMosaic authority.

## Current blocker before TR-POLY-09B

The Polyphonic V2 model has teacher-forced training/forward execution but no current free-running Polyphonic V2 inference contract.

A common benchmark must score predictions generated from image input without gold target prefixes. Therefore the next implementation should add bounded deterministic greedy decoding and strict V2 semantic validation before any comparative benchmark result is considered valid.

Minimum exit criteria:

```text
admitted image + exact checkpoint
        ↓
free-running greedy token generation
        ↓
EOS / bounded termination
        ↓
V2 detokenize + strict parse
        ↓
canonical prediction OR explicit invalid/abstain
        ↓
provenance-bound VALIDATION evidence
```

## Recommended next order

1. Freeze exact candidate/checkpoint/tokenizer/dataset/benchmark identities.
2. Implement bounded Polyphonic V2 free-running greedy inference.
3. Complete/admit deterministic benchmark metric implementations and adapters required by TR-POLY-02.
4. Run common VALIDATION benchmark on identical benchmark identities with separate 1/2/3/4+ voice strata.
5. Perform validation-only error decomposition and make only evidence-supported model/data changes.
6. Freeze final candidate, preprocessing, decoder, metrics and abstention policy.
7. Open sealed TEST once for the final Stage 9 decision.
8. If accepted, enter a separate ScoreMosaic shadow/integration gate; do not grant automatic production authority.

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

Implement the bounded Polyphonic V2 free-running inference/semantic-decoding contract, then enter TR-POLY-09B common VALIDATION benchmarking under the frozen TR-POLY-02 benchmark identity.
