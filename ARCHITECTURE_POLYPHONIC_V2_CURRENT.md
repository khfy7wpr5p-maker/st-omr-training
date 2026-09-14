# ST-OMR Training — Polyphonic V2 Current Architecture and Roadmap

Updated: 2026-09-14

This document is the detailed current-state companion to `ARCHITECTURE_CURRENT.md`. It separates implemented capability from measured quality and preserves the dependency order through the first common Polyphonic V2 benchmark.

## 1. Current baseline and active package

Protected-main baseline used to start the active package:

`2bb0ac23e11fb953bb3ada4aaef6369073b7e1f1`

Current chain:

```text
TR-POLY-02 evaluation taxonomy / benchmark identity       ✅
TR-POLY-03 external dataset + license registry            ✅
TR-POLY-04 deterministic external benchmark harness       ✅
TR-POLY-05 Polyphonic Representation V2                   ✅
TR-POLY-06 V2 parser / tokenizer / lossless roundtrip     ✅
TR-POLY-07 research model registry                         ✅
TR-POLY-08 tiny 2D Transformer prototype                  ✅
TR-POLY-08A bounded training + provenance                  ✅
TR-POLY-08B exact checkpoint persistence/reload            ✅
TR-POLY-08C exact Stage 6 V1→V2 dataset execution         ✅
TR-POLY-09A native explicit Polyphonic V2 materialization ✅
TR-POLY-09B1 free-running greedy inference                ✅ IMPLEMENTED / MERGE GATE
TR-POLY-09B2 deterministic metric/adaptor layer            🔄 NEXT
TR-POLY-09B3 common VALIDATION benchmark                   🔒
```

## 2. What is implemented versus what is proven

### Implemented

- explicit structured Polyphonic V2 target representation;
- exact onset/duration/staff/voice fields;
- deterministic canonical V2 JSON;
- lossless V2 tokenizer/detokenizer;
- 2D visual Transformer encoder preserving row × column patch memory;
- teacher-forced TRAIN/VALIDATION execution;
- deterministic training/checkpoint provenance;
- exact checkpoint persistence and reload verification;
- Stage 6 single-voice V1→V2 execution bridge;
- native explicit Polyphonic V2 dataset/materialization path;
- voice 2 / 3 / 4+ dataset-coverage gates;
- family leakage and sealed-TEST protections;
- BOS-only deterministic greedy free-running inference;
- one-time image encoding with autoregressive decoder-memory reuse;
- strict V2 semantic validation after EOS;
- explicit invalid/abstain states for malformed output, generated PAD/BOS and decode-limit exhaustion;
- exact inference/checkpoint identity binding.

### Not yet proven

- useful Polyphonic V2 recognition accuracy;
- superiority over V1 or specialist/hybrid candidates;
- reliable 2/3/4+ voice separation;
- relation quality for ties, beams, accidentals or cross-staff notation;
- scanned/phone robustness;
- sealed TEST performance;
- ScoreMosaic shadow readiness;
- production authority.

Implementation of inference closes a measurement prerequisite; it is not itself model-quality evidence.

## 3. TR-POLY-09B1 closes the teacher-forcing gap

Before B1, the 2D model could be scored only while receiving the true preceding target tokens. That is not valid end-to-end OMR evidence.

B1 freezes the first honest prediction path:

```text
verified image tensor
        ↓
full 2D visual memory
        ↓
BOS only
        ↓
argmax next token
        ↓
append prediction and repeat
        ↓
EOS / generated-control failure / max-step stop
        ↓
strict V2 parser
        ↓
canonical prediction OR explicit invalid/abstain
```

The image is encoded once; `decode_from_memory()` reuses the same full row × column memory for every autoregressive step.

### First-search policy

The initial common baseline is greedy by design:

- deterministic argmax;
- lowest token id wins a numerical tie under the pinned runtime behavior;
- no beam search;
- no sampling;
- no grammar-based logit masking;
- no semantic repair;
- no fabricated EOS;
- generated PAD or a second BOS is recorded as model failure;
- strict parser failure is recorded as semantic-invalid output.

This keeps model quality separate from search/repair quality. A later constrained or beam decoder must use a separately fingerprinted profile and be compared as its own candidate.

## 4. Candidate identity freeze now available

Every B1 result binds:

- exact model-state SHA-256;
- model-profile SHA-256;
- inference-profile SHA-256;
- max decode-step budget;
- tokenizer fingerprint;
- representation/tokenizer versions;
- pinned PyTorch runtime.

Verified-checkpoint inference additionally binds:

- checkpoint SHA-256;
- metadata file SHA-256;
- receipt SHA-256;
- metadata fingerprint;
- dataset manifest;
- preprocessing fingerprint;
- trainer profile;
- training provenance;
- model-registry record fingerprint;
- repository SHA.

This provides the model/checkpoint portion of P09B-0. The benchmark identity, metric-set identity and candidate adapter identity must still be frozen by B2/B3.

## 5. Next critical gap: deterministic metric/adaptor support

The next missing component is not another model architecture. It is a common scoring bridge that turns free-running prediction/reference pairs into the already-frozen TR-POLY-02 evaluation surface.

B2 should provide versioned, deterministic adapters for the metrics that can be computed from Polyphonic V2 without inventing unavailable information.

### Priority metric surface

#### Serialization / validity

- parse success;
- canonical V2 validity;
- MusicXML validity only where an admitted deterministic export adapter exists.

#### Sequence

- token error rate;
- normalized edit distance;
- exact sequence accuracy.

#### Musical semantic

- pitch accuracy;
- duration accuracy;
- onset accuracy;
- voice accuracy;
- staff accuracy.

#### Relations

Where both reference and prediction expose the relation deterministically:

- beam relation F1;
- tie relation F1;
- accidental-note F1;
- note-staff F1;
- notehead-stem F1 only if the representation/adaptor can define it without hidden visual labels.

#### Structural

TEDn remains reserved by TR-POLY-02 but must not be fabricated. It may enter only after the exact implementation/algorithm and license boundary is reviewed and versioned.

## 6. Common benchmark readiness checklist

TR-POLY-09B3 comparative execution stays closed until:

- B1 free-running inference is green on protected main;
- native V2 TRAIN/VALIDATION materialization remains hash-verifiable;
- compared checkpoint identities are frozen;
- the benchmark dataset/split identity is frozen;
- candidate adapters are versioned;
- metric implementations are deterministic and versioned;
- unsupported metrics are explicitly marked unsupported;
- TEST remains sealed.

## 7. Required benchmark strata

The first common VALIDATION report must not collapse all polyphony into one number.

At minimum report separately:

- `1_voice`;
- `2_voice`;
- `3_voice`;
- `4_plus_voice`.

Where admitted data exists, robustness buckets should also remain separate:

- clean;
- scan;
- phone;
- blur;
- perspective;
- low contrast.

Aggregate metrics may be added, but never as the only reported evidence.

## 8. Recommended package order

### P09B-1 — Free-running inference

Status: implemented in the active package; merge requires exact-head green CI.

Exit condition:

- a verified checkpoint consumes one admitted image;
- no gold target prefix is supplied;
- output becomes either canonical V2 or explicit invalid/abstain evidence;
- model state does not mutate;
- all relevant model/checkpoint/runtime identities are bound.

### P09B-2 — Metric implementation and candidate adapters

Status: next.

Exit condition:

- two compatible prediction/reference pairs can be scored by one versioned metric surface;
- candidate-specific interpretation is prohibited;
- unsupported fields remain unsupported;
- benchmark identity is included in the evidence binding;
- TEST remains unopened.

### P09B-3 — Common VALIDATION benchmark

Status: blocked until B2.

Exit condition:

- compared candidates use the same exact benchmark/split identity;
- per-voice-stratum reports exist;
- parse/sequence/semantic/relation evidence is separated;
- no TEST artifact is opened.

### P09C — Evidence-driven refinement

Use only TRAIN/VALIDATION evidence to determine the dominant failure family:

- visual resolution / patching;
- sequence search;
- representation length;
- data imbalance;
- voice separation;
- staff assignment;
- rhythm/onset modeling;
- relation modeling;
- robustness/domain coverage.

Only a demonstrated failure should justify expanding model, data or search complexity.

### P09D — Candidate freeze before sealed TEST

Freeze the complete recipe:

- model architecture;
- checkpoint;
- preprocessing;
- decoding algorithm;
- resource bounds;
- metric versions;
- benchmark adapter;
- abstention policy.

After this freeze, sealed TEST cannot become a tuning loop.

### Stage 9 — Sealed held-out decision

Run the already-frozen candidate once on held-out TEST evidence. Valid outcomes include accept, reject and insufficient evidence. A poor result does not authorize repeated TEST-driven tuning.

### Stage 10 — Separate ScoreMosaic shadow gate

A successful Stage 9 candidate still has no automatic production authority. Integration must separately verify artifact provenance, resource bounds, deterministic validator compatibility, failure/abstain behavior and regression safety.

## 9. Specialist/hybrid relationship

The Stage 7-D specialist lane remains historical and reusable evidence.

Potential later architecture:

```text
2D Polyphonic V2 recognizer
        +
local specialists / deterministic geometry
        ↓
versioned fusion candidate
        ↓
same common benchmark contract
```

Hybrid fusion should wait until the standalone free-running 2D candidate has a measurable baseline. Otherwise attribution of improvement/regression is impossible.

## 10. Data strategy after benchmark evidence

Do not default to “more data” or “larger model”. Use measured strata:

```text
VALIDATION error strata
        ↓
dominant failure family
        ↓
rights/provenance/split check
        ↓
targeted TRAIN/VALIDATION data or bounded model/search change
        ↓
new deterministic artifact identity
        ↓
retrain and re-evaluate
```

Examples:

- weak 4+ voice result → increase admitted high-polyphony evidence;
- weak cross-staff relation → add cross-staff-specific targets;
- strong clean / weak scan → improve admitted scan-domain coverage first;
- good token sequence / weak relations → improve relation modeling rather than visual capacity.

## 11. Do not do next

Until the common VALIDATION path is operational, do not prioritize:

- opening sealed TEST;
- ScoreMosaic/production integration;
- a larger Transformer by default;
- broad hyperparameter sweeps;
- beam/repair decoding before greedy evidence is frozen;
- automatic learning from teacher corrections;
- external data without rights/install-pin admission;
- presenting teacher-forced loss or implementation success as OMR accuracy.

## 12. Immediate next action

Finish the TR-POLY-09B1 exact-head CI/merge gate. Then implement **TR-POLY-09B2 deterministic metric/adaptor support** and proceed directly to **TR-POLY-09B3 common VALIDATION benchmarking** under the frozen TR-POLY-02 identity. TEST remains sealed throughout B1–B3.
