# ST-OMR Training — Polyphonic V2 Current Architecture and Roadmap

Updated: 2026-09-14

This document separates implemented capability from measured quality and preserves the dependency order toward a defensible Polyphonic V2 benchmark.

## Current chain

Protected `main` baseline:

`58c27b5403301fe478c9b6c8351682c8f8bf1624`

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
TR-POLY-09B2 deterministic metric/adaptor layer           ✅ IMPLEMENTED / MERGE GATE
TR-POLY-09B3 common VALIDATION aggregation                🔄 NEXT
missing metric admission                                  🔒 5 surfaces
P09C evidence-driven refinement                           🔒
P09D final candidate freeze                               🔒
Stage 9 sealed TEST decision                              🔒
Stage 10 ScoreMosaic shadow integration                   🔒
```

## Capability now implemented

```text
explicit canonical V2 target + image
        ↓
hash-bound TRAIN/VALIDATION materialization
        ↓
2D Transformer training/checkpoint path
        ↓
verified checkpoint
        ↓
BOS-only free-running greedy prediction
        ↓
strict V2 parse OR explicit invalid/abstain
        ↓
versioned deterministic metric adapter
```

B1 makes the candidate measurable without teacher forcing. B2 scores the exact metric semantics that can be supported from the currently frozen V2 representation without inventing relation/export structure.

## B2 exact metric coverage

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

- `musicxml_validity`: deterministic V2→MusicXML benchmark adapter not admitted;
- `tedn`: exact reviewed TEDn implementation not admitted;
- `notehead_stem_f1`: event-level stem direction is not an explicit notehead↔stem relation object;
- `beam_relation_f1`: event beam states do not provide an explicit cross-event beam relation identity;
- `tie_relation_f1`: notehead START/STOP states do not provide an explicit cross-event tie relation identity.

This is a deliberate fail-closed boundary. Related notation state is not promoted into a stronger frozen metric by approximation.

## B2 alignment and scoring policy

Sequence metrics use deterministic unit-cost Levenshtein over token surfaces after BOS.

Semantic fields use deterministic event alignment inside corresponding part/measure positions. Arbitrary IDs do not drive matching. Numeric semantic evidence covers pitch spelling, exact rational duration/onset, logical voice and event staff.

Exact relation metrics are limited to semantics explicitly represented on noteheads:

- pitch ↔ displayed accidental;
- pitch ↔ effective staff including `staff_override`.

Invalid/abstain free-running predictions stay in the report. They receive `parse_success=0`, sequence errors from actual generated tokens, zero on available semantic/relation metrics, while unsupported metrics remain unsupported.

## Identity and comparability

Each B2 sample report binds:

- sample SHA-256;
- BenchmarkIdentity SHA-256;
- B1 inference evidence SHA-256;
- reference V2 SHA-256;
- prediction V2 SHA-256 when valid;
- voice stratum;
- robustness bucket;
- metric/adaptor/alignment versions;
- ordered availability/value/reason state for every frozen metric ID.

## What B3 may and may not do

B3 can aggregate VALIDATION reports by:

- `1_voice`;
- `2_voice`;
- `3_voice`;
- `4_plus_voice`;
- robustness bucket where admitted.

It can produce useful partial common evidence for sequence, semantic, accidental and staff-association quality.

It must not claim a complete winner while five frozen metrics remain unsupported.

```text
partial common VALIDATION evidence     ✅ permitted
11 numeric frozen metrics               ✅ available
5 explicit unsupported metrics          ⚠️ visible
TR-POLY-02-complete metric record        🔒 unavailable
winner / promotion claim                 🔒 unavailable
sealed TEST                               🔒 closed
```

## B3 required architecture

```text
same exact BenchmarkIdentity
        +
VALIDATION descriptors
        +
B1 inference evidence
        +
B2 sample reports
        ↓
deterministic aggregate report
        ├─ global partial metrics
        ├─ 1/2/3/4+ voice strata
        ├─ robustness buckets
        ├─ invalid/abstain counts
        └─ available/unsupported metric coverage
```

Required behavior:

- VALIDATION only;
- benchmark identity equality enforced;
- duplicate sample identities rejected;
- invalid/abstain results retained;
- available metric means computed only from reports where that metric is numerically available, while coverage count is explicit;
- unsupported metrics propagated, never averaged as zero;
- canonical aggregate fingerprint;
- no winner/promotion field while metric coverage is incomplete;
- no TEST access.

## Missing-metric admission

Before a final complete TR-POLY-02 comparison, independently implement and review:

1. deterministic V2→MusicXML export plus independent validation;
2. exact notehead-stem relation representation/adapter if the metric remains required;
3. explicit cross-event beam relation representation/adapter;
4. explicit cross-event tie relation representation/adapter;
5. exact TEDn algorithm/dependency/license/version surface.

These packages should not modify candidate model weights or search behavior.

## Evidence-driven refinement after B3

Use measured VALIDATION strata to choose P09C work:

```text
high parse failure       -> decoder/search diagnosis
high TER, valid parses   -> sequence modeling/search diagnosis
weak pitch               -> visual/pitch evidence diagnosis
weak onset/duration      -> rhythm representation/training diagnosis
weak voice               -> polyphonic separation/data diagnosis
weak note_staff_f1       -> staff/cross-staff diagnosis
strong clean, weak scan  -> domain coverage diagnosis
```

Do not enlarge the model or broadly add data without such evidence.

## Final order

```text
B2 exact-head CI + merge
        ↓
B3 partial common VALIDATION aggregation
        ↓
exact admission of five missing metric surfaces
        ↓
TR-POLY-02-complete comparable evidence
        ↓
P09C evidence-driven TRAIN/VALIDATION refinement
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
- Unsupported frozen metrics cannot receive proxies under the same metric ID.
- Invalid/abstain outputs stay visible.
- External data requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- Benchmark success never grants automatic production authority.

## Immediate next action

Finish TR-POLY-09B2 exact-head CI/merge, then implement TR-POLY-09B3 deterministic VALIDATION aggregation/reporting with explicit metric coverage. Keep missing-metric, winner, and sealed-TEST gates closed until their independent evidence requirements are satisfied.
