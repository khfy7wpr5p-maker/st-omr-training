# ST-OMR Training — Current Architecture Overlay

Updated: 2026-09-14

This file records the active architecture lane. `ARCHITECTURE.md` remains the long-form historical record; `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` contains the detailed roadmap.

## Current baseline

- protected `main`: `79c2631682ebdb3b2be30c146a191e3ef8183ad0`
- latest merged package: PR #156 — TR-POLY-09B2 deterministic V2 metric/adaptor layer
- active package: TR-POLY-09B3 deterministic VALIDATION aggregation/reporting
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Active pipeline

```text
Polyphonic Representation V2                         ✅ FROZEN
        ↓
V2 parser / tokenizer / lossless roundtrip          ✅
        ↓
Tiny 2D Transformer + bounded trainer               ✅ RESEARCH
        ↓
Exact checkpoint persistence/reload                  ✅
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
Real checkpoint × VALIDATION execution               🔄 NEXT EXECUTION GATE
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

B3 does not create new model predictions. It consumes B2 sample reports and creates one deterministic candidate-level VALIDATION report.

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

### Frozen aggregation policy

`sample-macro-mean-v1`

For each available metric B3 records sample count, mean, minimum and maximum. It does not introduce hidden weighting by token count, score length, family size or note count.

Invalid/abstain outputs remain in the population because B2 emits them as scored sample reports rather than dropping them.

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

The required voice strata remain:

```text
1_voice
2_voice
3_voice
4_plus_voice
```

Missing strata are not silently ignored; they are recorded in `missing_voice_strata` and keep the common-comparison gate closed.

Observed robustness buckets are reported separately from the frozen set:

```text
clean
scan
phone
blur
perspective
low_contrast
```

## Metric coverage boundary

B2/B3 currently expose 11 numeric frozen metrics and five explicit unsupported metrics.

Available:

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

Unsupported:

```text
musicxml_validity
tedn
notehead_stem_f1
beam_relation_f1
tie_relation_f1
```

B3 preserves unsupported status and reasons. Unsupported metrics are never averaged as zero and never receive proxy values.

## Comparison boundary

B3 can validate whether multiple candidate reports are comparable, but it does not rank candidates.

A complete common-comparison gate requires:

- checkpoint-bound candidate evidence;
- all required voice strata;
- all frozen TR-POLY-02 metrics numerically admitted;
- identical benchmark identity and exact VALIDATION sample set across candidates.

Because five metrics remain unsupported, complete winner/promotion claims remain closed even after B3 infrastructure is green.

## Real measurement after B3

The next meaningful work is execution, not another model redesign:

```text
freeze checkpoint + VALIDATION manifest/build
        ↓
run B1 on every VALIDATION image
        ↓
produce one B2 report per sample
        ↓
aggregate through B3
        ↓
inspect measured failure strata
        ↓
choose P09C refinement only from evidence
```

This is where real pitch, duration, onset, voice, staff and robustness numbers will appear.

## Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN alone may update parameters.
- VALIDATION evaluation is read-only.
- Invalid/abstain outputs must remain in denominators.
- Unsupported metrics must not receive proxy numbers under frozen metric IDs.
- Different benchmark/candidate identities must never be mixed in one aggregate.
- External data still requires rights/license/install-pin admission.
- ScoreMosaic uploads and teacher corrections are not automatic training data.
- Candidate artifacts and reports remain hash/provenance bound.
- Deterministic musical validators retain veto authority.

## Next gate

Merge TR-POLY-09B3 only after exact-head CI is green. Then execute the frozen checkpoint against the admitted native V2 VALIDATION artifacts, produce B1→B2→B3 evidence, and use those measured strata to choose P09C. Do not open TEST or claim a complete benchmark winner while any frozen metric remains unsupported.
