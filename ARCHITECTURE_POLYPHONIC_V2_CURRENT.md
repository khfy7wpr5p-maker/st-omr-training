# ST-OMR Training — Polyphonic V2 Current Architecture and Roadmap

Updated: 2026-09-16

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
TR-POLY-09B7 hash-bound quality VALIDATION execution      ✅ MERGED
TR-POLY-09B8A persisted artifact reload/preflight         🔄 ACTIVE
first measured quality baseline                           ⛔ NEEDS ADMITTED REAL ARTIFACT ROOT
P09C evidence-driven refinement                           🔒
missing metric admission                                  🔒 5 surfaces
P09D final candidate/evaluation freeze                    🔒
Stage 9 sealed TEST decision                              🔒
Stage 10 ScoreMosaic shadow integration                   🔒
```

Latest merged package: PR #161 / `main` `a57a14a764e97640e89b6cd4d805254b7f2c1b2f`.
B7 exact PR head `2b9fcf9ac58dca7913dcb46e8117236f22ea2974` passed CI run #701 before merge.

## 2. Architecture layers

### Representation layer

Polyphonic Representation V2 and its tokenizer preserve explicit multi-voice score structure: voices, staves, chords, exact rational timing, noteheads, ties, beams, tuplets, grace state and cross-staff notehead overrides.

### Model layer

The tiny 2D Transformer keeps a 2D visual patch grid and autoregressive V2 decoder. It remains a research architecture with no production authority.

### Smoke layer

TR-POLY-08A/B prove bounded trainability and checkpoint replay only. Their ≤2-step semantics remain immutable.

### Quality-training layer

B4/B5/B6 form the separate genuine-candidate path:

```text
native V2 TRAIN/VALIDATION
        ↓
B6 deterministic selected-population materialization
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

B7 connects a verified B5 artifact to the frozen B1/B2/B3 stack:

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

### Persisted-artifact admission layer

B8A bridges externally persisted TR-POLY-09A roots back into the validated in-memory build object required by B6:

```text
persisted native V2 root
        ↓
canonical metadata + checksum verification
        ↓
exact TRAIN/VALIDATION artifact-set membership
        ↓
SHA-256 + V2 roundtrip + PNG semantic verification
        ↓
reconstructed NativePolyV2DatasetBuild
        ↓
B8A receipt
```

This is additive. It does not change the frozen TR-POLY-09A writer or the B6/B7 execution semantics.

## 3. B4 quality-training contract

B4 remains deliberately separate from the smoke trainer.

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

B5 defines a separate research artifact for a B4-selected state. It binds selected model state, B4 result fingerprint, exact model/trainer/provenance identity, dataset manifest, preprocessing fingerprint, tokenizer/representation/runtime, registry record and artifact hashes.

Reload verifies hashes before deserialization and uses `weights_only=True`. The B1 wrapper binds inference identity to the exact verified B5 artifact.

## 5. B6 native quality execution

B6 selects TRAIN/VALIDATION samples in deterministic sample-ID order, partitions the full selected population into contiguous batches of at most eight, verifies target/image bytes and canonical V2 roundtrip, forbids semantic truncation, keeps families disjoint, executes B4 and persists/verifies B5.

B6 never opens TEST and grants no benchmark or production authority.

## 6. B7 benchmark execution

B7 evaluates the **complete** native V2 VALIDATION population. There is no benchmark `max_samples`, random sampling or prefix selection.

Every explicit descriptor maps one-to-one to an exact VALIDATION sample and family ID. The split-manifest hash binds dataset/build identity, target/image/representation hashes, dimensions, target token count, complexity metadata and robustness bucket.

Before inference B7 verifies one B5 checkpoint, dataset identity, preprocessing/materialization identity, loaded model profile and one explicit bounded `max_decode_steps`. The checkpoint is loaded once and every sample runs frozen B1 greedy inference.

Invalid outputs remain in the population. B2/B3 semantics are unchanged.

## 7. B8A reload/preflight contract

A persisted root is accepted only if all of the following hold:

- root and metadata/artifact paths are regular non-symlink entries;
- `manifest.json` is exact canonical Native V2 JSON;
- `manifest.sha256` binds the exact manifest bytes;
- `build.json` matches the frozen builder/source/target/TEST policy;
- `targets/` and `images/` contain exactly the admitted TRAIN/VALIDATION hashes and no extra entries;
- each target hash, canonical V2 roundtrip, representation SHA, token count and polyphony profile matches the manifest;
- each image hash, grayscale PNG validity and dimensions match the manifest;
- reconstructed deterministic build ID matches `build.json`;
- the original TR-POLY-09A root verifier succeeds again.

The receipt records exact manifest/build identity and split sample IDs while fixing:

```text
test_artifact_bytes_accessed = false
production_authority = false
```

Extra artifacts are rejected before they can be admitted. TEST target/image bytes are not read.

## 8. First real quality baseline

B1–B7 are implemented, but implementation is not measured quality. The first meaningful run requires a physically available, admitted real Native V2 root. Repository synthetic fixtures remain regression evidence only.

Before execution freeze:

- dataset manifest SHA-256 and build ID from B8A;
- data-rights/license/install-pin evidence when external data is involved;
- exact B4 quality-training recipe;
- exact B6 batch/execution identity;
- exact B5 checkpoint hashes after training;
- explicit B7 descriptor manifest for the full VALIDATION split;
- exact B7 decode bound and benchmark identity.

Then:

```text
B8A verify real root
        ↓
B6 TRAIN-only candidate creation
        ↓
B5 checkpoint verify/reload
        ↓
B7 full VALIDATION execution
        ↓
B2/B3 failure analysis
        ↓
P09C targeted change
```

If an admitted root is not available, the baseline remains artifact-blocked. No quality number is estimated.

## 9. Measurement surface

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

No unsupported metric receives a proxy number. Full common-comparison readiness remains false until the missing surfaces are admitted exactly.

## 10. P09C decision policy

Use measured failure evidence rather than model-size intuition:

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

## 11. Evidence hierarchy

```text
SMOKE EVIDENCE
08A/08B: bounded plumbing/trainability

QUALITY TRAINING EVIDENCE
B4/B5/B6: multi-epoch candidate creation and verified artifact

QUALITY VALIDATION EVIDENCE
B7 + B1/B2/B3: exact checkpoint on exact hash-bound VALIDATION population

ARTIFACT ADMISSION EVIDENCE
B8A: persisted Native V2 root independently reconstructed and verified

SEALED TEST EVIDENCE
Stage 9 only: final one-shot decision
```

None automatically grants production authority.

## 12. Immediate next action

Finish B8A tests/docs and exact-head CI, and merge only if green. Then locate/provide the first admitted real Native V2 persisted root, freeze the experiment/B7 descriptor identities, execute B6→B5→B7, inspect actual VALIDATION failure strata and choose P09C from evidence. TEST remains sealed throughout.

## 13. Safety invariants

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
- No artifact reload, quality-training run or VALIDATION benchmark grants production authority.
- Every merge requires exact-head green CI.
