# ST-OMR Training — Polyphonic V2 Current Architecture and Roadmap

Updated: 2026-09-14

This document separates implemented capability from measured quality and preserves the dependency order toward a defensible Polyphonic V2 benchmark.

## Current chain

Protected `main` baseline:

`79c2631682ebdb3b2be30c146a191e3ef8183ad0`

```text
TR-POLY-02 evaluation taxonomy / benchmark identity       ✅
TR-POLY-03 external dataset + license registry            ✅
TR-POLY-04 deterministic external benchmark harness       ✅
TR-POLY-05 Polyphonic Representation V2                   ✅
TR-POLY-06 V2 parser/tokenizer/lossless roundtrip         ✅
TR-POLY-07 research model registry                         ✅
TR-POLY-08/08A/08B model + training + checkpoint          ✅ RESEARCH
TR-POLY-08C exact Stage 6 V1→V2 execution                 ✅
TR-POLY-09A native explicit V2 materialization            ✅
TR-POLY-09B1 free-running greedy inference                ✅ MERGED
TR-POLY-09B2 deterministic metric/adaptor layer           ✅ MERGED
TR-POLY-09B3 deterministic VALIDATION aggregation         ✅ IMPLEMENTED / MERGE GATE
real checkpoint × VALIDATION execution                    🔄 NEXT
P09C evidence-driven refinement                           🔒
missing metric admission                                  🔒 5 surfaces
P09D final candidate freeze                               🔒
Stage 9 sealed TEST decision                              🔒
Stage 10 ScoreMosaic shadow integration                   🔒
```

## End-to-end measurement architecture

```text
native V2 VALIDATION image + canonical target
        ↓
exact verified checkpoint
        ↓
B1 free-running greedy prediction
        ↓
strict V2 parse OR explicit invalid/abstain
        ↓
B2 deterministic per-sample metrics
        ↓
B3 deterministic candidate aggregation
        ├─ overall
        ├─ 1 / 2 / 3 / 4+ voice
        └─ robustness buckets
```

B1 removes teacher forcing. B2 provides honest per-sample metric evidence. B3 makes those sample reports comparable and inspectable without hiding failure cases.

## B3 aggregation contract

B3 accepts only B2 reports that are:

- VALIDATION-only;
- checkpoint-bound;
- from one exact benchmark identity;
- from one exact candidate identity;
- from the same versioned B2 metric/alignment/relation contract.

It rejects mixed benchmark identities, mixed candidate identities, duplicate sample IDs, unbound checkpoints and unexpected strata/buckets.

### Frozen first aggregation policy

`sample-macro-mean-v1`

Every available metric is averaged equally over samples. B3 records mean/min/max plus coverage counts. It does not silently weight longer scores or larger families more heavily.

A later pooled/micro policy, if needed, must receive its own version and evidence identity.

## Failure visibility

Free-running invalid/abstain outputs remain in the B3 population because B2 already assigns them explicit sample reports.

Therefore:

```text
parse-invalid sample
        ≠ dropped sample
        = visible failed sample in aggregate evidence
```

This prevents survivorship bias in parse success and available semantic metrics.

## Voice strata

Required voice coverage remains:

- `1_voice`
- `2_voice`
- `3_voice`
- `4_plus_voice`

B3 creates slices for observed strata and records any missing required strata explicitly. Missing voice coverage keeps the common-comparison gate closed.

## Robustness buckets

Observed buckets are kept separate:

- `clean`
- `scan`
- `phone`
- `blur`
- `perspective`
- `low_contrast`

A global average may be reported but cannot replace the per-stratum evidence.

## Current metric coverage

### Available numeric metrics — 11

- `parse_success`
- `ter`
- `normalized_edit_distance`
- `exact_sequence_accuracy`
- `pitch_accuracy`
- `duration_accuracy`
- `onset_accuracy`
- `voice_accuracy`
- `staff_accuracy`
- `accidental_note_f1`
- `note_staff_f1`

### Unsupported frozen metrics — 5

- `musicxml_validity`
- `tedn`
- `notehead_stem_f1`
- `beam_relation_f1`
- `tie_relation_f1`

B3 propagates these as unsupported with explicit reasons and no numeric proxy.

## Common comparison gate

B3 exposes comparability validation but no winner selection.

A candidate report becomes `common_comparison_ready` only when:

1. the candidate is checkpoint-bound;
2. all required voice strata are represented;
3. every frozen TR-POLY-02 metric is numerically admitted.

Two or more candidate reports may then be considered comparable only when they also share the exact benchmark identity and exact VALIDATION sample IDs.

Today this gate remains closed because five metrics are unsupported. Partial diagnostic VALIDATION evidence is still permitted and is sufficient to guide P09C refinement.

## Real execution after B3 merge

B3 infrastructure alone does not produce real model quality numbers. The next execution package/run must bind the actual external artifacts:

```text
exact native V2 VALIDATION manifest/build
        +
exact verified checkpoint
        ↓
for each VALIDATION sample:
    image → B1 → B2 report
        ↓
all sample reports → B3 aggregate
        ↓
measured failure strata
```

The resulting report should answer:

- Does the model parse reliably?
- How large is token error?
- Are pitches correct?
- Are durations/onsets correct?
- Does voice separation collapse as polyphony increases?
- Are notes assigned to the correct staff?
- Are scan/phone/blur domains weaker than clean data?

## P09C decision policy

Use measured evidence rather than model-size intuition:

```text
high parse failure       → decoder/search diagnosis
high TER, valid parses   → sequence modeling/search diagnosis
weak pitch               → visual/pitch evidence diagnosis
weak duration/onset      → rhythm modeling/data diagnosis
weak 2+/3+/4+ voice      → polyphonic separation/data diagnosis
weak note_staff_f1       → staff/cross-staff diagnosis
strong clean, weak scan  → robustness/domain coverage diagnosis
```

Only demonstrated failure families should justify model, data or search expansion.

## Missing-metric admission

Before a complete TR-POLY-02 winner/promotion claim, separately implement and review:

1. deterministic V2→MusicXML export plus independent validation;
2. exact notehead-stem relation representation/adapter if retained as required;
3. explicit cross-event beam relation representation/adapter;
4. explicit cross-event tie relation representation/adapter;
5. exact TEDn algorithm/dependency/license/version surface.

These packages should not modify model weights merely to complete the reporting surface.

## Final order

```text
B3 exact-head CI + merge
        ↓
real B1→B2→B3 VALIDATION execution
        ↓
P09C evidence-driven TRAIN/VALIDATION refinement
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
- TRAIN is the only parameter-updating split.
- VALIDATION is read-only evidence.
- Unsupported metrics cannot receive proxy values under frozen IDs.
- Invalid/abstain outputs remain visible in denominators.
- Different benchmark/candidate identities cannot be mixed.
- External data requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- Benchmark evidence never grants automatic production authority.

## Immediate next action

Finish TR-POLY-09B3 exact-head CI/merge. Then execute the exact checkpoint over the admitted native V2 VALIDATION artifacts and aggregate the resulting B1/B2 evidence through B3. Use those measured strata—not assumptions—to choose P09C refinement. TEST remains sealed.
