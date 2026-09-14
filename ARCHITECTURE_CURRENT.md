# ST-OMR Training — Current Architecture Overlay

Updated: 2026-09-14

This file records the active architecture lane. `ARCHITECTURE.md` remains the long-form historical record; `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` contains the detailed Polyphonic V2 roadmap.

## Current baseline

- latest merged package: PR #160 — TR-POLY-09B6 native multi-batch quality execution
- latest verified pre-B7 `main`: `429e64d790d758c8f99ad143e0a2a04c878c04bc`
- active package: TR-POLY-09B7 hash-bound quality VALIDATION execution
- frozen ≤2-step smoke trainer/checkpoint: preserved unchanged
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Active pipeline

```text
Polyphonic Representation V2                            ✅ FROZEN
        ↓
V2 parser / tokenizer / lossless roundtrip             ✅
        ↓
Tiny 2D Transformer                                    ✅ RESEARCH
        ↓
≤2-step smoke trainer + smoke checkpoint               ✅ FROZEN
        ↓
Native explicit V2 TRAIN/VALIDATION materialization    ✅ 09A
        ↓
B1 BOS-only free-running inference                     ✅
        ↓
B2 deterministic per-sample metrics                    ✅
        ↓
B3 deterministic VALIDATION aggregation                ✅
        ↓
B4 multi-epoch TRAIN-only quality training             ✅
        ↓
B5 verified quality-checkpoint + B1 binding            ✅
        ↓
B6 full native multi-batch TRAIN/VALIDATION execution  ✅
        ↓
B7 descriptor-bound quality VALIDATION execution       🔄 ACTIVE
        ↓
first measured quality baseline                        🔒
        ↓
P09C evidence-driven refinement                        🔒
        ↓
missing metric admission (5 surfaces)                  🔒
        ↓
P09D candidate/evaluation freeze                       🔒
        ↓
Stage 9 one-shot sealed TEST                           🔒
        ↓
Stage 10 ScoreMosaic shadow/integration                🔒
```

## What B4–B6 changed

The historical Poly2D path remains a smoke proof capped at two optimizer steps. Quality work was added as a separate lane instead of weakening those semantics.

```text
B4
exact TRAIN batches + read-only VALIDATION
→ deterministic multi-epoch optimization
→ minimum mean VALIDATION-loss selected state

B5
selected B4 state
→ separate non-overwriting artifact schema
→ hash verification before weights-only reload
→ checkpoint-bound B1 identity

B6
full selected native TRAIN/VALIDATION population
→ deterministic sample-id order
→ contiguous batches, each ≤8 samples
→ B4 training
→ B5 checkpoint
```

B6 never opens TEST, never truncates an overlong semantic target and never grants benchmark or production authority.

## B7 closes the execution bridge

B1/B2/B3 already defined how quality is measured, but there was no single execution boundary proving that an exact B5 quality checkpoint was evaluated against one exact descriptor-bound VALIDATION population.

B7 provides that bridge:

```text
exact native V2 VALIDATION sample/artifact identity
        +
explicit complexity + robustness descriptor
        ↓
canonical split_manifest_sha256
        ↓
TR-POLY-02 BenchmarkIdentity
        +
verified B5 checkpoint
        ↓
B1 free-running inference
        ↓
B2 sample metrics
        ↓
B3 aggregate report
```

### B7 population policy

- complete native V2 VALIDATION population only;
- no `max_samples` benchmark escape hatch;
- no random/prefix selection;
- one descriptor per sample;
- descriptor family identity must equal native manifest family identity;
- TEST descriptor rejected before dataset-root/checkpoint access.

### B7 descriptor policy

Complexity and robustness values are explicit metadata. They are not silently inferred where TR-POLY-02 has no frozen derivation algorithm.

In particular, absence of robustness provenance does not become `clean` automatically.

The split-manifest SHA binds every declared descriptor to exact target/image/representation hashes, dimensions and target token count. Metadata drift therefore changes the benchmark identity.

### B7 checkpoint/inference policy

- load and independently verify one B5 checkpoint;
- require checkpoint dataset manifest == benchmark dataset manifest;
- require checkpoint preprocess/materialization identity == current native materialization;
- require loaded model profile == checkpoint model profile;
- freeze one explicit `max_decode_steps` for the whole execution;
- load checkpoint once and run B1 read-only for every VALIDATION sample;
- bind every B1 result to the same checkpoint/metadata/receipt/provenance identity;
- reject any candidate identity drift inside the run.

## Measurement semantics

Invalid B1 outputs are not filtered. B2 turns invalid/abstain predictions into explicit parse failure plus sequence/semantic evidence, and B3 keeps them in the macro population.

Current numeric metrics:

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

Still unsupported:

```text
musicxml_validity
tedn
notehead_stem_f1
beam_relation_f1
tie_relation_f1
```

No proxy may inherit any unsupported frozen metric ID.

## Claim boundary

A green B7 merge means the repository can produce genuine, hash-bound VALIDATION evidence from a verified quality checkpoint. It still does not mean:

- a sufficiently large/admitted quality corpus has already been executed;
- the model is good enough for production;
- full TR-POLY-02 metric coverage exists;
- a candidate wins a comparison;
- TEST may be opened;
- ScoreMosaic integration is authorized.

## Required order from here

```text
B7 exact-head CI + merge
        ↓
freeze admitted real/native TRAIN/VALIDATION build
        +
freeze explicit B7 descriptor manifest
        ↓
B6 train + persist exact quality checkpoint
        ↓
B7 complete VALIDATION execution
        ↓
inspect actual B2/B3 failure strata
        ↓
P09C targeted refinement
        ↓
admit missing metrics exactly
        ↓
P09D freeze
        ↓
Stage 9 sealed TEST once
        ↓
Stage 10 shadow integration
```

## Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN is the only split allowed to update parameters.
- VALIDATION is read-only.
- Frozen smoke semantics remain immutable.
- Semantic target truncation is forbidden.
- Invalid/abstain predictions remain in benchmark denominators.
- Unsupported metrics remain nonnumeric.
- External data requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- No training or VALIDATION benchmark artifact grants production authority.
