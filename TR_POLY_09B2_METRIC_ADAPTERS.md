# TR-POLY-09B2 — Deterministic Polyphonic V2 Metric Adapters

Status: implementation package; common benchmark execution and sealed TEST remain closed.

## Purpose

TR-POLY-09B1 established free-running Polyphonic V2 predictions without gold target prefixes. TR-POLY-09B2 adds the deterministic scoring bridge from one VALIDATION reference/prediction pair into the frozen TR-POLY-02 metric vocabulary.

This package deliberately distinguishes **implemented metrics** from **declared-but-not-yet-admitted metrics**. It does not fabricate numeric values for metric surfaces that have no accepted implementation.

## Admitted input boundary

The metric adapter consumes exactly:

```text
canonical PolyScore reference
        +
TR-POLY-09B1 free-running inference result
        +
frozen BenchmarkIdentity
        +
VALIDATION BenchmarkSampleDescriptor
        ↓
versioned PolyV2SampleMetricReport
```

TRAIN and TEST descriptors are rejected. The package opens no dataset bytes itself and authorizes no TEST access.

## Metric status

### Implemented

Serialization / prediction validity:

- `parse_success`

Sequence:

- `ter`
- `normalized_edit_distance`
- `exact_sequence_accuracy`

Musical semantic:

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

### Explicitly unsupported in B2

- `musicxml_validity` — no admitted deterministic Polyphonic V2 → MusicXML export/validation adapter is yet bound to this benchmark surface;
- `tedn` — TR-POLY-02 reserves the metric name, but no exact TEDn algorithm/version/license implementation has yet been admitted in this repository.

Unsupported metrics remain present in the report with `availability=unsupported`, no numeric value and an explicit reason. Therefore the report cannot be promoted to a complete TR-POLY-02 metric result while either surface remains unsupported.

## Sequence policy

Sequence comparison operates on the frozen V2 token IDs excluding the initial BOS. Unit-cost Levenshtein distance is deterministic.

Definitions:

```text
TER = edits / reference-token-count
NED = edits / max(reference-token-count, prediction-token-count)
exact_sequence_accuracy = 1 iff complete token-id tuples are equal, otherwise 0
```

Free-running invalid/abstain predictions remain in the evaluation population. They are not dropped from reports.

## Event alignment

Semantic and relation metrics use a versioned unit-cost deterministic event alignment inside corresponding part/measure positions.

The alignment signature excludes arbitrary event/atom IDs and uses musical content such as:

- event kind;
- onset and duration;
- logical voice and staff;
- visible note type;
- pitch surface;
- dots/stem;
- beam/tuplet/grace metadata;
- tie, accidental and cross-staff information.

Substitution is preferred on deterministic dynamic-programming ties before deletion/insertion.

## Semantic metrics

Semantic accuracies are reference-oriented. A missing predicted event therefore contributes zero correctness for the corresponding reference event.

Pitch is compared as the sorted event notehead spelling surface `(step, alter, octave)` rather than by arbitrary note IDs.

Invalid/unparseable free-running outputs receive:

- `parse_success = 0`;
- sequence edit metrics from the actual generated token sequence;
- semantic metrics = 0;
- relation metrics = 0.

This prevents invalid outputs from disappearing from aggregate benchmark evidence.

## Relation metrics

Relations are represented as deterministic multisets inside aligned events and scored by standard F1 over exact relation identities.

Examples:

- notehead ↔ stem direction;
- beam level/state;
- pitch ↔ tie state;
- pitch ↔ visible accidental intent;
- pitch ↔ effective staff, including `staff_override`.

An empty reference and empty prediction relation set yields F1 = 1.0. Extra or missing relations are penalized through false positives/false negatives.

## Evidence identity

Every sample report binds:

- sample SHA-256;
- benchmark identity SHA-256;
- B1 inference evidence SHA-256;
- reference representation SHA-256;
- prediction representation SHA-256 when parse-valid;
- voice stratum;
- robustness bucket;
- metric adapter version;
- event-alignment version;
- relation-metric version;
- complete ordered metric availability/value surface.

Changing the benchmark identity changes the report fingerprint even when prediction/reference content is unchanged.

## Full-contract admission gate

`require_complete_required_metrics()` is deliberately fail-closed.

It can succeed only when every frozen TR-POLY-02 metric is numerically available and then passes `validate_required_metric_result(...)`.

With B2 alone, it remains closed because `musicxml_validity` and `tedn` are unsupported.

This distinction is important:

```text
B2 partial/common metric evidence    ✅ available
TR-POLY-02-complete metric record    🔒 not yet available
benchmark winner claim               🔒 prohibited
```

## Regression coverage

Tests cover:

- exact prediction → perfect implemented metrics;
- explicit unsupported MusicXML/TEDn surfaces;
- full-contract gate refusal while unsupported metrics remain;
- field-specific pitch/onset/voice errors;
- relation-specific stem/beam/tie/accidental/staff errors;
- invalid free-running output retained with zero semantic/relation credit;
- deterministic report fingerprint;
- benchmark-identity binding;
- TRAIN and TEST descriptor rejection.

## Safety / non-claims

TR-POLY-09B2 does not:

- open TEST;
- train or mutate any model;
- select or promote a checkpoint;
- implement or approximate TEDn;
- fabricate MusicXML validity;
- claim a benchmark winner;
- modify ScoreMosaic;
- grant production authority.

## Exit condition

The package may merge only after exact-head CI is green.

A green merge means that free-running V2 evidence can be scored deterministically on the currently admitted subset of the frozen TR-POLY-02 metric surface. It does not mean the metric contract is complete or that the recognizer is accurate.

## Next gate

TR-POLY-09B3 should add a common VALIDATION execution/report harness that:

- consumes one exact BenchmarkIdentity;
- aggregates B2 sample reports by 1/2/3/4+ voice strata and robustness bucket;
- preserves invalid/abstain samples in denominators;
- exposes metric coverage/unsupported counts;
- refuses a complete winner/promotion decision while required metrics remain unsupported;
- keeps TEST sealed.

MusicXML validity and TEDn admission may be implemented as separate, independently reviewed packages before any final TR-POLY-02-complete comparison claim.
