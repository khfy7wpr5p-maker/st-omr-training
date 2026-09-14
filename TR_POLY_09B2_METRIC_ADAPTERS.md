# TR-POLY-09B2 — Deterministic Polyphonic V2 Metric Adapters

Status: implementation package; common benchmark execution and sealed TEST remain closed.

## Purpose

TR-POLY-09B1 established free-running Polyphonic V2 predictions without gold target prefixes. TR-POLY-09B2 adds the deterministic scoring bridge from one VALIDATION reference/prediction pair into the frozen TR-POLY-02 metric vocabulary.

The package distinguishes **exactly representable metrics** from metric names whose required relation/export structure is not present yet. It never fills an unsupported metric with a proxy number.

## Input boundary

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

TRAIN and TEST descriptors are rejected. B2 opens no dataset bytes itself and authorizes no TEST access.

## Metric status

### Available numeric metrics

Prediction validity:

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

Relations explicitly represented by V2 notehead data:

- `accidental_note_f1`
- `note_staff_f1`

### Explicitly unsupported in B2

- `musicxml_validity` — no admitted deterministic Polyphonic V2 → MusicXML benchmark export/validation adapter;
- `tedn` — no exact reviewed TEDn implementation/version/license surface admitted;
- `notehead_stem_f1` — V2 stores event-level stem direction, not an explicit notehead↔stem relation object;
- `beam_relation_f1` — V2 stores per-event beam marks, not an explicit cross-event beam relation identity;
- `tie_relation_f1` — V2 stores START/STOP tie state on noteheads, not an explicit cross-event tie relation identity.

Those states remain useful semantics, but treating them as the stronger frozen relation-F1 metrics would silently redefine TR-POLY-02. B2 refuses that shortcut.

Every unsupported metric remains in the ordered report with `availability=unsupported`, no numeric value, and an explicit reason.

## Sequence policy

Sequence comparison uses frozen V2 token IDs after BOS and deterministic unit-cost Levenshtein distance.

```text
TER = edits / reference-token-count
NED = edits / max(reference-token-count, prediction-token-count)
exact_sequence_accuracy = 1 iff complete token-id tuples are equal
```

Invalid/abstain free-running outputs remain in the evaluation population.

## Event alignment and semantic metrics

Semantic fields use a versioned deterministic event alignment inside corresponding part/measure positions. Arbitrary event/atom IDs do not drive matching.

Reference-oriented numeric semantics are:

- sorted pitch spelling surface `(step, alter, octave)`;
- exact rational duration;
- exact rational onset;
- logical voice;
- event staff.

Missing predicted reference events receive zero correctness.

For invalid/unparseable free-running output:

- `parse_success = 0`;
- sequence metrics use the actual generated token sequence;
- implemented semantic metrics = 0;
- implemented relation metrics = 0;
- unsupported metrics remain unsupported rather than becoming zero.

## Exact relation metrics

`accidental_note_f1` compares explicit `(pitch, display_accidental)` associations.

`note_staff_f1` compares explicit `(pitch, effective_staff)` associations, including `staff_override` for cross-staff placement.

They are scored with deterministic multiset F1 inside aligned events. An empty reference and empty prediction relation set yields F1 = 1.0; extra or missing relations contribute false positives/false negatives.

## Evidence identity

Every sample report binds:

- sample SHA-256;
- benchmark identity SHA-256;
- B1 inference evidence SHA-256;
- reference V2 SHA-256;
- prediction V2 SHA-256 when parse-valid;
- voice stratum;
- robustness bucket;
- metric-adapter version;
- event-alignment version;
- relation-metric version;
- every frozen metric ID with its availability/value/reason state.

Changing benchmark identity changes report identity even when musical content is unchanged.

## Full-contract gate

`require_complete_required_metrics()` fails closed until every frozen TR-POLY-02 metric is numerically admitted.

Current state:

```text
B2 partial/common metric evidence    ✅ available
11 exact numeric metrics             ✅ available
5 frozen metrics                     ⚠️ explicit UNSUPPORTED
TR-POLY-02-complete metric record    🔒 unavailable
winner / promotion claim             🔒 prohibited
```

## Regression coverage

Tests verify:

- exact prediction → perfect exactly representable metrics;
- all five unsupported surfaces are explicit and non-numeric;
- stem/beam/tie states do not become proxy relation scores;
- field-specific pitch/onset/voice errors remain visible;
- explicit accidental/staff relation errors are scored;
- invalid output remains in available metrics rather than being excluded;
- deterministic report identity and benchmark binding;
- TRAIN/TEST descriptor rejection;
- full-contract admission refusal while unsupported metrics remain.

## Safety / non-claims

TR-POLY-09B2 does not:

- open TEST;
- train or mutate a model;
- select/promote a checkpoint;
- approximate TEDn under the TEDn metric ID;
- fabricate MusicXML validity;
- reinterpret stem/beam/tie state as an explicit relation metric;
- claim a complete benchmark winner;
- modify ScoreMosaic;
- grant production authority.

## Next gate

After exact-head CI and merge, TR-POLY-09B3 should aggregate B2 VALIDATION sample reports by 1/2/3/4+ voice strata and robustness bucket, preserve invalid/abstain samples, expose metric coverage, and refuse any complete winner/promotion field while frozen metrics remain unsupported.

MusicXML validity, notehead-stem relation, beam relation, tie relation and TEDn can be admitted later only through separate exact/versioned evidence surfaces.
