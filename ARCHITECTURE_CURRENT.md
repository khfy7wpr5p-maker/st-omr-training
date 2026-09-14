# ST-OMR Training — Current Architecture Overlay

Updated: 2026-09-14

This file records the current active lane. `ARCHITECTURE.md` remains the long-form historical architecture record. The detailed Polyphonic V2 roadmap is in `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md`.

## Current repository baseline

- protected branch baseline for TR-POLY-09B1: `main` at `2bb0ac23e11fb953bb3ada4aaef6369073b7e1f1`
- latest previously merged technical package: PR #153 — `TR-POLY-09A: native Polyphonic V2 dataset materialization`
- architecture documentation refresh: PR #154 — merged
- active implementation package: TR-POLY-09B1 bounded free-running Polyphonic V2 greedy inference
- sealed TEST: remains closed
- ScoreMosaic / production authority: not granted

## Active pipeline

```text
Canonical / explicit notation targets
        ↓
Deterministic symbolic validation + rendering
        ↓
Hash-bound image/target artifacts
        ↓
Polyphonic Representation V2                    ✅ FROZEN
        ↓
V2 parser + tokenizer + lossless roundtrip       ✅ CLOSED
        ↓
Research model registry                           ✅ CLOSED
        ↓
Tiny 2D Transformer architecture                  ✅ IMPLEMENTED / RESEARCH
        ↓
Bounded TRAIN-only trainer + VALIDATION read-only ✅ IMPLEMENTED
        ↓
Exact checkpoint persistence/reload               ✅ IMPLEMENTED
        ↓
Exact Stage 6 V1→V2 execution bridge              ✅ CLOSED / single-voice evidence only
        ↓
Native explicit Polyphonic V2 dataset path        ✅ CLOSED — TR-POLY-09A
        ↓
Free-running V2 greedy inference                  ✅ IMPLEMENTED — TR-POLY-09B1
        ↓
Strict V2 semantic validation / explicit abstain  ✅ IMPLEMENTED — TR-POLY-09B1
        ↓
TR-POLY-09B2 metric/adaptor implementation        🔄 NEXT
        ↓
TR-POLY-09B3 common VALIDATION benchmark          🔒
        ↓
Polyphonic error-strata diagnosis/refinement       🔒
        ↓
Frozen candidate + sealed TEST decision            🔒 TEST SEALED
        ↓
Separate ScoreMosaic shadow/integration gate       🔒
```

## What TR-POLY-09A established

TR-POLY-09A added a separate native Polyphonic V2 dataset boundary without relabeling the frozen V1 corpus.

The admitted path is:

```text
explicit canonical V2 target + grayscale PNG
        ↓
native V2 manifest/build SHA-256 identity
        ↓
TRAIN/VALIDATION artifact hash verification
        ↓
lossless V2 JSON/token roundtrip
        ↓
deterministic image preprocessing
        ↓
Poly2DTrainingBatch
        ↓
TR-POLY-08A bounded trainer
        ↓
TR-POLY-08B checkpoint/reload
```

The native dataset contract requires explicit polyphonic evidence including voice 2 and corpus-level coverage of voice 3, voice 4+, note/rest/chord, simultaneous independent voices, and chord-vs-independent-voice same-onset cases. Missing voice/onset/duration/staff information is not inferred.

TR-POLY-09A does **not** establish benchmark success, quality improvement, production readiness, ScoreMosaic readiness, or a winner over existing baselines.

## TR-POLY-09B1 architecture delta

Before this package, the 2D Transformer exposed only a teacher-forced forward path. That was insufficient for end-to-end OMR comparison because the decoder received the gold target prefix.

TR-POLY-09B1 adds the missing measurement prerequisite:

```text
admitted image tensor
        ↓
2D visual encoder — once per image
        ↓
full row × column visual memory
        ↓
BOS-only prefix
        ↓
greedy next-token loop using decode_from_memory()
        ↓
EOS / invalid control token / max-step termination
        ↓
strict V2 detokenize + parser
        ↓
canonical PolyScore OR explicit invalid/abstain evidence
```

### Compatibility rule

`decode_from_memory()` is additive. It does not alter model parameters, state-dict layout, model configuration identity, tokenizer vocabulary or Polyphonic Representation V2. Existing exact checkpoint artifacts therefore remain structurally compatible.

### Frozen first-search policy

The first inference baseline deliberately uses deterministic greedy argmax only.

- BOS is the only initial token.
- EOS ends the sequence.
- generated PAD or a second BOS fails closed rather than being silently masked.
- exhausting the decode bound does not fabricate EOS.
- EOS output is accepted only if strict V2 reconstruction succeeds.
- malformed semantics produce explicit invalid/abstain evidence.
- model state must remain byte-identical before/after inference.
- v1 inference is one-image-at-a-time and CPU-bound to the pinned deterministic runtime.

Beam search, constrained repair and sampling remain separate future candidates. They must not be mixed into the first greedy benchmark baseline.

## Candidate identity now available for measurement

A verified-checkpoint inference result binds the exact:

- model-state SHA-256;
- model profile;
- inference profile and max-step policy;
- tokenizer fingerprint and representation version;
- checkpoint / metadata / receipt SHA-256 values;
- checkpoint metadata fingerprint;
- dataset-manifest identity;
- preprocessing fingerprint;
- trainer profile;
- training provenance;
- registry record fingerprint;
- repository SHA;
- pinned PyTorch runtime.

This closes the model-side P09B-0 identity requirement. The benchmark dataset/split identity and metric implementation identity are still owned by P09B-2/P09B-3.

## Recommended development order

```text
P09B-1  Free-running Polyphonic V2 inference
        ✅ implementation package
        - greedy BOS-only generation
        - strict semantic parse
        - explicit abstain/failure states
        - checkpoint/provenance binding

P09B-2  Deterministic metric implementations/adapters
        🔄 NEXT
        - connect free-running prediction/reference pairs to TR-POLY-02
        - sequence + semantic + relation surfaces first
        - structural metric such as TEDn only after implementation/license review
        - unsupported metrics remain unsupported, never invented

P09B-3  Common VALIDATION benchmark
        🔒
        - same frozen benchmark identity for compared candidates
        - 1 / 2 / 3 / 4+ voice strata
        - robustness buckets where admitted
        - report parse/sequence/semantic/relation results separately

P09C    Validation-only error decomposition and bounded refinement
        🔒
        - pitch / duration / onset / voice / staff
        - chord grouping / tie / beam / accidental association
        - model/data expansion only where benchmark evidence identifies a failure

P09D    Freeze final candidate and evaluation recipe
        🔒
        - architecture + preprocessing + decoder + checkpoint + metrics fixed
        - sealed TEST cannot become a tuning loop

Stage 9 sealed TEST candidate decision
        🔒 TEST SEALED
        - one-shot held-out evidence
        - accept / reject / insufficient-evidence

Stage 10 separate ScoreMosaic shadow integration
        🔒
        - no automatic production authority
        - deterministic validators retain veto authority
```

## Existing specialist lane

The Stage 7-D specialist work remains valid historical evidence and may later contribute to hybrid/fusion candidates. It is not discarded and must not be silently rewritten as current benchmark evidence.

A future hybrid must enter the same frozen benchmark identity and metric surface as the standalone Polyphonic V2 recognizer.

## Safety boundaries that remain unchanged

- TEST remains sealed until the final Stage 9 decision gate.
- TRAIN alone may update model parameters.
- VALIDATION is read-only evaluation/tuning evidence.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- External datasets require the existing rights/license/install-pin contract.
- Large datasets and checkpoints remain outside ordinary Git content.
- Every candidate artifact remains hash/provenance bound.
- Deterministic musical validators retain veto authority over learned outputs.
- No benchmark result may be presented as production accuracy unless its exact dataset/split/metric identity is stated.

## Next gate

After TR-POLY-09B1 is green on protected main, implement **TR-POLY-09B2 deterministic metric/adaptor support** against free-running V2 predictions. Then execute **TR-POLY-09B3 common VALIDATION benchmarking**. Do not open sealed TEST, enlarge the model, or add search heuristics before the first measurable greedy baseline exists.
