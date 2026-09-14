# ST-OMR Training — Current Architecture Overlay

Updated: 2026-09-14

This file records the active architecture lane. `ARCHITECTURE.md` remains the long-form historical record; `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` contains the detailed roadmap.

## Current baseline

- protected `main`: `79c2631682ebdb3b2be30c146a191e3ef8183ad0`
- latest merged package: PR #156 — TR-POLY-09B2 deterministic V2 metric/adaptor layer
- active package: TR-POLY-09B3 deterministic VALIDATION aggregation/reporting
- current Poly2D trainer: bounded smoke harness, maximum 2 update steps
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
Bounded ≤2-step smoke trainer + checkpoint          ✅ INFRASTRUCTURE
        ↓
Native explicit Polyphonic V2 TRAIN/VALIDATION      ✅ TR-POLY-09A
        ↓
BOS-only free-running greedy inference               ✅ TR-POLY-09B1
        ↓
Strict V2 parse / explicit invalid-abstain evidence ✅ TR-POLY-09B1
        ↓
Deterministic per-sample metric/adaptor surface      ✅ TR-POLY-09B2
        ↓
Deterministic VALIDATION aggregation                 ✅ IMPLEMENTED — TR-POLY-09B3
        ↓
Smoke checkpoint VALIDATION sanity run               🔄 OPTIONAL NEXT EXECUTION
        ↓
Versioned quality-training regime                    🔒 REQUIRED FOR QUALITY CLAIMS
        ↓
First quality-trained checkpoint                     🔒
        ↓
Real quality VALIDATION B1→B2→B3                    🔒
        ↓
Evidence-driven P09C refinement                      🔒
        ↓
Exact missing-metric admission                       🔒 5 surfaces
        ↓
Candidate freeze + sealed TEST decision              🔒 TEST SEALED
        ↓
Separate ScoreMosaic shadow/integration              🔒
```

## What B3 adds

B3 consumes B2 sample reports and creates one deterministic candidate-level VALIDATION report.

```text
checkpoint-bound B2 sample reports
        ↓
exact benchmark identity equality
        ↓
exact candidate identity equality
        ↓
duplicate/split/version checks
        ↓
sample-macro aggregation
        ├─ overall
        ├─ 1 / 2 / 3 / 4+ voice
        ├─ robustness buckets
        ├─ parse-success coverage
        └─ unsupported-metric coverage
```

The first aggregation policy is frozen as `sample-macro-mean-v1`. For each available metric B3 records sample count, mean, minimum and maximum. It does not introduce hidden weighting by token count, score length, family size or note count.

Invalid/abstain outputs remain in the population because B2 emits them as scored sample reports rather than dropping them.

## Training-readiness boundary

The current `Poly2DTrainingConfig` caps training at two smoke steps. Therefore existing checkpoint support demonstrates:

- deterministic parameter updates;
- TRAIN/VALIDATION split enforcement;
- checkpoint persistence and reload;
- provenance binding;
- inference and metric compatibility.

It does **not** demonstrate a converged or quality-trained OMR candidate.

Accordingly, B3 has two distinct uses:

1. **sanity baseline now** — run the current smoke checkpoint through B1→B2→B3 to validate the end-to-end measurement path;
2. **quality benchmark later** — first implement a separately versioned TRAIN-only multi-step/epoch training regime, train/freeze a candidate, then run the same B1→B2→B3 path on VALIDATION.

Smoke-checkpoint numbers must never be presented as the architecture's expected quality ceiling.

## Identity and leakage protections

B3 requires every sample report to be:

- `split=validation`;
- checkpoint-bound;
- from the same exact BenchmarkIdentity;
- from the same exact candidate identity;
- produced by the frozen B2 adapter/alignment/relation versions.

Mixed candidates, mixed benchmarks, duplicate samples, TRAIN/TEST evidence and unexpected strata are rejected.

The final B3 fingerprint binds exact sorted sample IDs and exact B2 sample-report fingerprints, so input ordering cannot alter evidence identity.

## Voice and robustness reporting

Required voice strata:

```text
1_voice
2_voice
3_voice
4_plus_voice
```

Missing strata are recorded in `missing_voice_strata` and keep the common-comparison gate closed.

Observed robustness buckets are reported separately:

```text
clean
scan
phone
blur
perspective
low_contrast
```

## Metric coverage boundary

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

Explicitly unsupported:

```text
musicxml_validity
tedn
notehead_stem_f1
beam_relation_f1
tie_relation_f1
```

Unsupported metrics are never averaged as zero and never receive proxy values.

## Comparison boundary

B3 can validate comparability but does not rank candidates. A complete common-comparison gate requires checkpoint-bound evidence, all required voice strata, all frozen TR-POLY-02 metrics numerically admitted, and identical benchmark/sample identities across candidates.

Because five metrics remain unsupported, complete winner/promotion claims remain closed.

## Recommended development order

```text
B3 exact-head CI + merge
        ↓
optional smoke B1→B2→B3 sanity baseline
        ↓
versioned quality-training regime (TRAIN only)
        ↓
train + freeze first quality candidate
        ↓
quality VALIDATION B1→B2→B3
        ↓
P09C evidence-driven refinement
        ↓
missing metric admission / complete comparison surface
        ↓
P09D candidate + evaluation recipe freeze
        ↓
Stage 9 one-shot sealed TEST decision
        ↓
Stage 10 independent ScoreMosaic shadow gate
```

## Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN alone may update parameters.
- VALIDATION evaluation is read-only.
- Invalid/abstain outputs remain in denominators.
- Unsupported metrics cannot receive proxies under frozen IDs.
- Different benchmark/candidate identities cannot be mixed.
- Smoke evidence cannot be relabeled as quality-trained evidence.
- External data still requires rights/license/install-pin admission.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Candidate artifacts and reports remain hash/provenance bound.
- Deterministic musical validators retain veto authority.

## Next gate

Merge TR-POLY-09B3 only after exact-head CI is green. Then either run the current checkpoint strictly as a sanity baseline or proceed directly to a separately versioned quality-training regime. A model-quality VALIDATION claim requires the latter. TEST remains sealed.
