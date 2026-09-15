# ST-OMR Training — Polyphonic V2 Current Architecture and Roadmap

Updated: 2026-09-14

This document separates implemented capability, executable evidence and measured model quality. TEST remains sealed until Stage 9.

## 1. Current chain

```text
TR-POLY-02 evaluation taxonomy / benchmark identity       ✅ FROZEN
TR-POLY-03 external dataset + license registry            ✅
TR-POLY-04 deterministic external benchmark harness       ✅
TR-POLY-05 Polyphonic Representation V2                   ✅ FROZEN
TR-POLY-06 V2 parser/tokenizer/lossless roundtrip         ✅ FROZEN
TR-POLY-07 research model registry                        ✅
TR-POLY-08 tiny 2D Transformer                            ✅ RESEARCH
TR-POLY-08A bounded trainer                               ✅ SMOKE ONLY (≤2 steps)
TR-POLY-08B bounded research checkpoint                   ✅ SMOKE ONLY
TR-POLY-08C exact Stage 6 V1→V2 execution                ✅ SINGLE-VOICE EVIDENCE
TR-POLY-09A native explicit V2 materialization            ✅ MERGED
TR-POLY-09B1 free-running greedy inference                ✅ MERGED
TR-POLY-09B2 deterministic metric/adaptor layer           ✅ MERGED
TR-POLY-09B3 deterministic VALIDATION aggregation         ✅ MERGED
TR-POLY-09B4 multi-epoch quality-training regime          ✅ MERGED
TR-POLY-09B5 verified quality checkpoint + B1 bridge      ✅ MERGED
TR-POLY-09B6 native multi-batch quality execution         ✅ MERGED
TR-POLY-09B7 hash-bound quality VALIDATION execution      🔄 ACTIVE
first measured quality baseline                           🔒
P09C evidence-driven refinement                           🔒
missing metric admission                                  🔒 5 surfaces
P09D final candidate/evaluation freeze                    🔒
Stage 9 sealed TEST decision                              🔒
Stage 10 ScoreMosaic shadow integration                   🔒
```

Latest merged package before B7: PR #160 / main `429e64d790d758c8f99ad143e0a2a04c878c04bc`.

## 2. Architecture layers

### Representation layer

Polyphonic Representation V2 and its tokenizer preserve the explicit score surface needed for multi-voice OMR research: voices, staves, chords, exact rational timing, noteheads, ties, beams, tuplets, grace state and cross-staff notehead overrides.

### Model layer

The tiny 2D Transformer keeps a 2D visual patch grid and autoregressive V2 decoder. It remains a research architecture with no production authority.

### Smoke layer

TR-POLY-08A/B prove bounded trainability and checkpoint replay only. Their ≤2-step semantics remain immutable.

### Quality-training layer

B4/B5/B6 form a separate path for genuine training candidates:

```text
native V2 TRAIN/VALIDATION
        ↓
B6 deterministic complete selected population materialization
        ↓
ordered batches, each ≤8
        ↓
B4 multi-epoch TRAIN updates + read-only VALIDATION
        ↓
minimum mean VALIDATION-loss selected epoch
        ↓
B5 hash-verified research quality checkpoint
```

### Quality-measurement layer

B7 connects a verified B5 artifact to the already-frozen B1/B2/B3 measurement stack:

```text
explicit VALIDATION descriptors
        +
exact native sample/artifact identities
        ↓
hash-bound split manifest
        ↓
TR-POLY-02 BenchmarkIdentity
        +
verified B5 checkpoint
        ↓
B1 free-running predictions
        ↓
B2 sample reports
        ↓
B3 aggregate report
```

## 3. B4 quality-training contract

B4 is deliberately separate from the smoke trainer.

Frozen v1 policies:

- TRAIN-only gradient updates;
- exact caller-supplied batch order;
- no shuffle or hidden sampling;
- no scheduler;
- full read-only VALIDATION after every epoch;
- fixed epoch budget;
- arithmetic-mean VALIDATION loss;
- minimum mean VALIDATION-loss selection;
- earliest epoch wins exact ties;
- selected state reloaded and hash-verified;
- TEST prohibited;
- production authority prohibited.

Safety ceilings remain finite: max 128 epochs, max 100,000 optimizer steps and inherited per-batch tensor/token limits.

## 4. B5 quality-checkpoint contract

B5 does not widen TR-POLY-08B. It defines a separate research artifact for a B4-selected state.

It binds:

- selected model state;
- B4 result fingerprint and selected epoch;
- exact model/trainer/provenance identity;
- dataset manifest;
- preprocessing fingerprint;
- tokenizer/representation/runtime;
- dedicated research registry record;
- checkpoint/metadata/training-result/receipt hashes.

Reload verifies hashes before deserialization and uses `weights_only=True`. The B1 wrapper binds inference identity to the exact verified B5 artifact.

## 5. B6 native quality execution

B6 removes the old execution limitation where native V2 materialization could feed only one ≤8-sample smoke batch.

B6:

- preserves the old smoke API unchanged;
- selects TRAIN or VALIDATION samples in deterministic sample-id order;
- supports the full selected population;
- partitions it into contiguous batches of at most eight;
- verifies target/image bytes and canonical V2 roundtrip;
- forbids semantic target truncation;
- keeps TRAIN/VALIDATION families disjoint;
- executes B4;
- persists/verifies B5;
- never opens TEST;
- grants no benchmark or production authority.

A green B6 merge proves the execution path, not model quality.

## 6. B7 benchmark population identity

B7 evaluates the **complete** native V2 VALIDATION population. Benchmark execution exposes no `max_samples` or random sampling option.

Every explicit descriptor must map one-to-one to an exact VALIDATION sample and exact family ID.

The B7 split-manifest hash binds:

- dataset manifest and native build ID;
- sample/family IDs;
- target SHA-256;
- canonical representation SHA-256;
- image SHA-256;
- dimensions and target-token count;
- caller-declared complexity profile;
- caller-declared robustness bucket;
- B7 descriptor/selection policy versions.

This SHA becomes `BenchmarkIdentity.split_manifest_sha256`.

## 7. Why descriptor values are not guessed

TR-POLY-02 freezes fields such as voice count, density measures and robustness buckets, but it does not define one universal extraction algorithm for all density/robustness metadata.

B7 therefore treats them as explicit benchmark metadata rather than inventing values.

Consequences:

- no absent robustness label becomes `clean` silently;
- no density is estimated by an unversioned helper;
- changing declared metadata changes benchmark identity;
- future automatic descriptor derivation requires its own versioned deterministic admission contract.

## 8. B7 checkpoint and inference gate

Before sample inference B7 verifies:

1. the complete B5 artifact through the existing loader;
2. checkpoint dataset manifest == B7 native dataset manifest;
3. checkpoint preprocessing identity == native V2 materialization fingerprint for the loaded model config;
4. checkpoint model-profile fingerprint == loaded model config;
5. one explicit `max_decode_steps` within the checkpoint target boundary.

The checkpoint is loaded once. Every sample then uses the frozen B1 greedy decoder without gold prefix or model mutation.

The first inference establishes candidate/inference-profile identity. Any identity drift on a later sample aborts the execution.

## 9. B2/B3 semantics remain unchanged

B7 orchestrates; it does not redefine metrics.

Invalid free-running predictions remain in the population. B2 records parse failure and available sequence/semantic relation values according to its frozen contract. B3 requires one benchmark identity, one checkpoint-bound candidate identity and unique sample IDs.

Voice strata:

- `1_voice`;
- `2_voice`;
- `3_voice`;
- `4_plus_voice`.

Robustness buckets:

- `clean`;
- `scan`;
- `phone`;
- `blur`;
- `perspective`;
- `low_contrast`.

Missing strata remain visible rather than fabricated.

## 10. Current metric coverage

Available numeric metrics:

- `parse_success`;
- `ter`;
- `normalized_edit_distance`;
- `exact_sequence_accuracy`;
- `pitch_accuracy`;
- `duration_accuracy`;
- `onset_accuracy`;
- `voice_accuracy`;
- `staff_accuracy`;
- `accidental_note_f1`;
- `note_staff_f1`.

Still explicitly unsupported:

- `musicxml_validity`;
- `tedn`;
- `notehead_stem_f1`;
- `beam_relation_f1`;
- `tie_relation_f1`.

A B7 run can therefore be genuine VALIDATION evidence while full common-comparison/promotion readiness remains false.

## 11. Evidence hierarchy

The architecture now distinguishes four evidence levels:

```text
SMOKE EVIDENCE
08A/08B: bounded plumbing/trainability

QUALITY TRAINING EVIDENCE
B4/B5/B6: real multi-epoch candidate creation and verified artifact

QUALITY VALIDATION EVIDENCE
B7 + B1/B2/B3: exact checkpoint on exact hash-bound VALIDATION population

SEALED TEST EVIDENCE
Stage 9 only: final one-shot decision
```

None of these automatically grants production authority.

## 12. First real quality baseline

After B7 is merged, the first meaningful model-quality run should freeze all of the following before execution:

- admitted native/external TRAIN/VALIDATION dataset identity;
- data-rights/license/install-pin evidence where external data is involved;
- exact B4 quality-training recipe;
- exact B6 batch/execution identity;
- exact B5 checkpoint hashes;
- explicit B7 descriptor manifest;
- exact B7 decode bound and benchmark identity.

Then:

```text
B6 TRAIN-only candidate creation
        ↓
freeze B5 checkpoint
        ↓
B7 full VALIDATION execution
        ↓
B2/B3 failure analysis
        ↓
P09C targeted change
```

The first run is a baseline, not a winner or production candidate.

## 13. P09C decision policy

Use measured evidence rather than model-size intuition:

```text
high parse failure       → decoder/search diagnosis
high TER, valid parses   → sequence/search diagnosis
weak pitch               → visual/pitch evidence diagnosis
weak duration/onset      → rhythm/data diagnosis
weak 2+/3+/4+ voice      → polyphonic separation/data diagnosis
weak note_staff_f1       → staff/cross-staff diagnosis
strong clean, weak scan  → robustness/domain coverage diagnosis
```

Only demonstrated failure families justify architecture/data/search expansion.

## 14. Missing metric admission

The five unsupported surfaces must be admitted separately and exactly. In particular:

- TEDn must use an identified/versioned implementation, not a private proxy with the same metric ID;
- MusicXML validity requires an admitted V2→MusicXML evaluation adapter;
- stem/beam/tie relation metrics require relation identities that match the frozen metric semantics.

Until then no complete winner/promotion claim is valid.

## 15. Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN is the only parameter-updating split.
- VALIDATION is read-only.
- Frozen smoke contracts remain immutable.
- Semantic target truncation is forbidden.
- Benchmark sample selection cannot silently shrink the population.
- Descriptor metadata cannot silently default or drift.
- Invalid/abstain outputs remain in denominators.
- Unsupported metric IDs never receive proxy values.
- Different benchmark/candidate identities cannot be mixed.
- External data requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- No quality-training or VALIDATION benchmark artifact grants production authority.

## Immediate next action

Finish B7 exact-head tests/CI and merge only if green. Then execute the first admitted B6→B5→B7 quality baseline, inspect actual VALIDATION failure strata, and choose P09C from evidence. TEST remains sealed throughout.
