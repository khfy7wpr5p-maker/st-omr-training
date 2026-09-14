# ST-OMR Training — Polyphonic V2 Current Architecture and Roadmap

Updated: 2026-09-14

This document is the detailed current-state companion to `ARCHITECTURE_CURRENT.md`. It describes what is already implemented on protected `main`, what is still missing, and the recommended dependency order for the next architecture packages.

## 1. Current protected-main baseline

Reviewed `main` head:

`6a6bf1e1faf0149ebd097a310536076b77feac65`

Latest merged package:

PR #153 — `TR-POLY-09A: native Polyphonic V2 dataset materialization`

The current Polyphonic V2 chain on main is:

```text
TR-POLY-02 evaluation taxonomy / benchmark identity      ✅
TR-POLY-03 external dataset + license registry           ✅
TR-POLY-04 deterministic external benchmark harness      ✅
TR-POLY-05 Polyphonic Representation V2                  ✅
TR-POLY-06 V2 parser / tokenizer / lossless roundtrip    ✅
TR-POLY-07 research model registry                        ✅
TR-POLY-08 tiny 2D Transformer prototype                 ✅
TR-POLY-08A bounded training + provenance                ✅
TR-POLY-08B exact checkpoint persistence/reload           ✅
TR-POLY-08C exact Stage 6 V1→V2 dataset execution        ✅
TR-POLY-09A native explicit Polyphonic V2 materialization✅
TR-POLY-09B common benchmark                              🔒 NEXT
```

## 2. What is implemented versus what is proven

### Implemented

- explicit structured polyphonic target representation;
- exact onset/duration/staff/voice fields;
- deterministic canonical V2 JSON;
- lossless V2 tokenizer/detokenizer;
- 2D visual Transformer encoder preserving row × column patch memory;
- autoregressive decoder architecture under teacher forcing;
- TRAIN-only optimizer path;
- read-only VALIDATION path;
- deterministic provenance fingerprints;
- exact checkpoint hash/reload verification;
- Stage 6 single-voice V1→V2 execution path;
- native explicit Polyphonic V2 dataset/materialization path;
- voice 2 / 3 / 4+ coverage gates at the dataset contract level;
- family leakage and sealed-TEST protections.

### Not yet proven

- free-running Polyphonic V2 recognition quality;
- greedy or beam inference correctness on real benchmark samples;
- common benchmark superiority versus V1 baseline or specialist/hybrid candidates;
- robust relation recovery for ties, beams, accidentals and cross-staff cases;
- production-quality scanned/phone-photo performance;
- ScoreMosaic shadow readiness;
- production authority.

## 3. Critical architecture gap before comparison

The current 2D model exposes a teacher-forced training/forward surface, but there is no corresponding Polyphonic V2 free-running inference contract yet.

That matters because a valid OMR benchmark must score outputs produced from the image without feeding the gold target prefix. Teacher-forced token loss or teacher-forced logits are useful training diagnostics, not end-to-end recognition evidence.

The first missing runtime component is therefore:

```text
verified image tensor
        ↓
2D Transformer visual memory
        ↓
BOS-only decoder start
        ↓
autoregressive next-token loop
        ↓
EOS / max-step termination
        ↓
V2 detokenize + strict parser
        ↓
semantic-valid V2 output OR explicit invalid/abstain result
```

## 4. Required inference contract

The initial inference surface should be deliberately narrow.

### First admitted decoder

Use deterministic greedy decoding first.

Requirements:

- one exact model/checkpoint identity;
- one exact tokenizer fingerprint;
- BOS required at start;
- EOS terminates the sequence;
- PAD is never generated as meaningful content;
- maximum decode-step bound;
- exact vocabulary bounds;
- finite logits only;
- deterministic same-input/same-checkpoint/same-runtime result;
- strict V2 detokenization and canonical parser validation;
- no silent insertion of missing onset/duration/voice/staff values;
- invalid semantic output becomes explicit failure/abstention evidence;
- no TEST access.

### Why beam search should wait

Beam search can improve sequence metrics but changes the search algorithm itself. If introduced before a greedy baseline is frozen, model quality and search quality become confounded.

Recommended sequence:

1. freeze greedy decode evidence;
2. run common VALIDATION benchmark;
3. only then evaluate beam/constrained search as a separately fingerprinted candidate if needed.

## 5. Benchmark readiness checklist

TR-POLY-09B should not start comparative scoring until all of the following are true:

- native V2 TRAIN/VALIDATION materialization is hash-verifiable;
- candidate checkpoint can be reloaded exactly;
- free-running V2 inference exists;
- benchmark identity is fixed;
- compared candidates use the same dataset/split identity where comparison is claimed;
- metric implementations are versioned and deterministic;
- unavailable metrics are marked unsupported rather than fabricated;
- TEST remains sealed.

## 6. Common benchmark surface

The benchmark should preserve the TR-POLY-02 hierarchy.

### Serialization

- parse success;
- MusicXML validity where a deterministic export adapter exists.

### Sequence

- token error rate;
- normalized edit distance;
- exact sequence accuracy.

### Structural

- TEDn or another explicitly versioned structural distance only after implementation and license review.

### Musical semantic

- pitch accuracy;
- duration accuracy;
- onset accuracy;
- voice accuracy;
- staff accuracy.

### Relations

- notehead-stem F1;
- beam relation F1;
- tie relation F1;
- accidental-note F1;
- note-staff F1.

### Required strata

At minimum report separately:

- 1 voice;
- 2 voices;
- 3 voices;
- 4+ voices.

Where data exists, also keep robustness buckets such as clean, scan, phone, blur, perspective and low contrast distinct. Do not collapse all results into one aggregate number only.

## 7. Recommended next package order

These labels express the recommended dependency order; they are not claims that the packages already exist.

### P09B-0 — Candidate freeze

Freeze:

- repository SHA;
- candidate registry row;
- checkpoint SHA;
- model profile;
- trainer profile;
- tokenizer fingerprint;
- representation version;
- preprocessing fingerprint;
- native V2 dataset manifest/build identity;
- benchmark identity;
- runtime fingerprint.

No comparison should proceed if those identities drift mid-run.

### P09B-1 — Polyphonic V2 inference

Implement deterministic free-running greedy decode and strict semantic validation.

Exit condition: a reloaded checkpoint can consume an admitted image and produce either a canonical parseable V2 prediction or an explicit failure/abstain result with deterministic provenance.

### P09B-2 — Metric implementation/adapters

Complete any still-missing deterministic metric code and candidate adapters required by TR-POLY-02.

Exit condition: two compatible prediction/reference pairs can be scored through the same versioned metric surface without candidate-specific interpretation.

### P09B-3 — Common VALIDATION benchmark

Run comparable candidates against the same frozen VALIDATION benchmark identity.

Exit condition: per-stratum reports exist for sequence, semantic, structural and relation metrics; no TEST bytes are opened.

### P09C — Evidence-driven refinement

Use only TRAIN/VALIDATION evidence to determine whether the dominant failure is:

- visual resolution / patching;
- sequence search;
- representation length;
- data imbalance;
- voice separation;
- staff assignment;
- rhythm/onset modeling;
- relation modeling;
- missing robustness coverage.

Only the demonstrated failure mode should justify architecture expansion.

### P09D — Candidate freeze before sealed TEST

Freeze the complete evaluation recipe:

- model architecture;
- checkpoint;
- preprocessing;
- decoding algorithm;
- thresholds;
- metric versions;
- benchmark adapter;
- abstention policy.

After this freeze, sealed TEST must not become a tuning loop.

### Stage 9 — Sealed held-out decision

Use the held-out TEST only for the final candidate decision under the already-frozen recipe.

Possible outcomes should include acceptance, rejection or insufficient-evidence/abstain. A poor TEST result must not trigger repeated tuning against TEST.

### Stage 10 — Separate ScoreMosaic shadow gate

Even a successful sealed benchmark does not directly grant production authority.

A later integration gate should independently verify:

- artifact provenance;
- runtime resource bounds;
- failure/abstain behavior;
- deterministic validator compatibility;
- no silent score mutation;
- regression against accepted historical OMR surfaces.

## 8. Specialist/hybrid relationship

The Stage 7-D specialist line is not discarded.

Potential future architecture:

```text
2D Polyphonic V2 recognizer
        +
local specialists / deterministic geometry
        ↓
versioned fusion candidate
        ↓
same common benchmark contract
```

However, hybrid fusion should be added only after the standalone 2D model has a measurable benchmark baseline. Otherwise it becomes impossible to determine which component caused an improvement or regression.

## 9. Data strategy after TR-POLY-09A

The immediate priority is not simply “more data”. Data expansion should follow benchmark error evidence.

Preferred decision logic:

```text
benchmark error strata
        ↓
identify dominant under-covered failure family
        ↓
check rights / provenance / split safety
        ↓
add explicit TRAIN/VALIDATION data for that family
        ↓
rebuild deterministic manifest identity
        ↓
retrain / re-evaluate
```

Examples:

- weak 4+ voice score → increase explicit high-polyphony examples;
- weak cross-staff relation → add cross-staff-specific admitted data;
- strong clean / weak scan → improve admitted scan-domain coverage rather than enlarging the model first;
- good localization / poor relation scores → improve relation-target modeling before adding visual capacity.

## 10. Do not do next

Until the common benchmark path is operational, do not prioritize:

- opening sealed TEST;
- production/ScoreMosaic integration;
- a larger Transformer by default;
- broad hyperparameter sweeps;
- automatic learning from teacher corrections;
- mixing external datasets without rights/install-pin admission;
- treating teacher-forced loss as end-to-end OMR accuracy.

## 11. Immediate next action

Implement P09B-1: the bounded free-running Polyphonic V2 greedy inference + strict semantic validation surface. Once that path is green and provenance-bound, proceed directly to the frozen common VALIDATION benchmark under TR-POLY-02.
