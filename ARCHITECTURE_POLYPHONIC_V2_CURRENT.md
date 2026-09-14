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
TR-POLY-09B-complete metric admission                     🔒 MusicXML validity + TEDn pending
P09C evidence-driven refinement                           🔒
P09D final candidate freeze                               🔒
Stage 9 sealed TEST decision                              🔒
Stage 10 ScoreMosaic shadow integration                   🔒
```

## Capability now implemented

The research path now contains all of the following surfaces:

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

B1 means the candidate is measurable without teacher forcing. B2 means those free-running outputs can be compared to canonical V2 references on a substantial, versioned portion of the frozen TR-POLY-02 metric vocabulary.

Neither package proves that the recognizer is accurate.

## B2 metric coverage

### Available numeric metrics

Serialization:

- `parse_success`

Sequence:

- `ter`
- `normalized_edit_distance`
- `exact_sequence_accuracy`

Musical semantics:

- `pitch_accuracy`
- `duration_accuracy`
- `onset_accuracy`
- `voice_accuracy`
- `staff_accuracy`

Relations:

- `notehead_stem_f1`
- `beam_relation_f1`
- `tie_relation_f1`
- `accidental_note_f1`
- `note_staff_f1`

### Frozen metrics still unsupported

`musicxml_validity`

Reason: no admitted deterministic V2 → MusicXML export/validation adapter is bound to this benchmark path yet.

`tedn`

Reason: TR-POLY-02 reserves TEDn, but the exact algorithm/version/license implementation has not yet been admitted. B2 does not approximate TEDn under the frozen metric name.

Every sample report contains both unavailable metric IDs with explicit unsupported reasons and no numeric value.

## Why incomplete metrics do not block diagnostic benchmarking

B3 can still produce useful VALIDATION evidence for the implemented metric families. This is enough to diagnose whether the dominant problem is sequence generation, pitch, rhythm/onset, voice separation, staff assignment or relation recovery.

However, B3 must distinguish:

```text
partial common VALIDATION evidence     ✅ permitted
TR-POLY-02-complete result             🔒 unavailable
winner / promotion claim               🔒 unavailable
sealed TEST opening                     🔒 unavailable
```

The partial benchmark must never be described as the complete frozen benchmark.

## B2 alignment and scoring policy

Sequence metrics use deterministic unit-cost Levenshtein over V2 token surfaces after BOS.

Semantic and relation metrics use a versioned unit-cost event alignment inside corresponding part/measure positions. Arbitrary event and notehead IDs are excluded from the alignment signature.

Reference-oriented semantic correctness measures:

- pitch surface `(step, alter, octave)`;
- exact rational duration;
- exact rational onset;
- logical voice;
- event staff.

Relations are exact multiset identities within aligned events:

- pitch ↔ stem direction;
- beam level/state;
- pitch ↔ tie state;
- pitch ↔ displayed accidental;
- pitch ↔ effective staff, including cross-staff override.

Invalid/abstain free-running predictions remain benchmark evidence. They receive `parse_success=0`, their real generated token sequence contributes sequence edits, and semantic/relation metrics receive zero credit rather than being excluded.

## Identity and comparability

A B2 report binds:

- sample SHA-256;
- BenchmarkIdentity SHA-256;
- B1 inference evidence SHA-256;
- reference V2 SHA-256;
- prediction V2 SHA-256 when valid;
- voice stratum;
- robustness bucket;
- metric-adapter version;
- event-alignment version;
- relation-metric version;
- ordered metric availability/value surface.

Changing the benchmark identity changes report identity even for identical prediction/reference content.

## B3 next architecture

B3 should be an aggregation/execution layer, not a new model package.

```text
same frozen BenchmarkIdentity
        +
VALIDATION BenchmarkSampleDescriptor set
        +
B1 inference results
        +
B2 sample metric reports
        ↓
deterministic aggregate report
        ├─ 1_voice
        ├─ 2_voice
        ├─ 3_voice
        ├─ 4_plus_voice
        └─ robustness buckets
```

Required B3 behavior:

- VALIDATION only;
- exact benchmark identity equality across all samples/candidates;
- no dropping invalid/abstain outputs;
- metric coverage counts alongside means;
- explicit unsupported metric propagation;
- deterministic canonical report fingerprint;
- no winner/promotion field while the required metric set is incomplete;
- TEST bytes never opened.

## After B3

Use B3 evidence to select only demonstrated failure families for P09C.

Examples:

```text
high parse failure          -> decoder/search/sequence grammar diagnosis
high TER, good parse        -> sequence model/search diagnosis
weak pitch only             -> visual/pitch representation diagnosis
weak onset/duration         -> rhythmic representation/training diagnosis
weak voice                  -> polyphonic separation/data diagnosis
weak note_staff_f1          -> staff/cross-staff diagnosis
strong clean / weak scan    -> domain-coverage diagnosis
```

Do not enlarge the model or expand data broadly without this evidence.

## Metric-completion packages

Before a final complete comparison claim, independently admit:

1. deterministic Polyphonic V2 → MusicXML export + independent MusicXML validation for `musicxml_validity`;
2. exact TEDn implementation with version, algorithm semantics, dependency/license review and deterministic regression evidence.

These should remain separable from model tuning so metric implementation cannot silently change candidate behavior.

## Final dependency order

```text
B2 exact-head CI + merge
        ↓
B3 partial common VALIDATION aggregation
        ↓
MusicXML validity + TEDn metric admission
        ↓
TR-POLY-02-complete comparable evidence
        ↓
P09C evidence-driven refinement on TRAIN/VALIDATION
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
- VALIDATION evaluation is read-only.
- Unsupported metrics remain unsupported; no proxy inherits a frozen metric ID.
- Invalid/abstain outputs stay in denominators.
- External data requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- Benchmark success never grants automatic production authority.

## Immediate next action

Finish TR-POLY-09B2 exact-head CI/merge, then implement TR-POLY-09B3 deterministic VALIDATION aggregation/reporting by voice stratum and robustness bucket. Keep complete-winner and sealed-TEST gates closed until the missing required metrics are independently admitted.
