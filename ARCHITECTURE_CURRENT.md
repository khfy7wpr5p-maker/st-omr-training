# ST-OMR Training — Current Architecture Overlay

Updated: 2026-09-16

This file records the active architecture lane. `ARCHITECTURE.md` remains the long-form historical record; `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` contains the detailed Polyphonic V2 roadmap.

## Current baseline

- latest merged package: PR #161 — TR-POLY-09B7 hash-bound quality VALIDATION execution
- latest verified `main`: `a57a14a764e97640e89b6cd4d805254b7f2c1b2f`
- B7 exact PR head `2b9fcf9ac58dca7913dcb46e8117236f22ea2974`: CI run #701 success
- active package: TR-POLY-09B8A persisted Native V2 artifact reload / first-baseline preflight
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
B7 descriptor-bound quality VALIDATION execution       ✅ MERGED
        ↓
B8A persisted artifact reload / baseline preflight     🔄 ACTIVE
        ↓
first measured quality baseline                        ⛔ NEEDS ADMITTED REAL ARTIFACT ROOT
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

## B4–B7 quality path

The historical Poly2D path remains a smoke proof capped at two optimizer steps. Quality work is a separate lane and does not weaken those semantics.

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

B7
complete exact VALIDATION population
+ explicit descriptor metadata
+ verified B5 checkpoint
→ B1 free-running inference
→ B2 per-sample metrics
→ B3 aggregate report
```

B7 is now a merged execution capability. It still does not establish a measured model-quality baseline until an admitted real corpus is actually run.

## B8A closes the persisted-artifact handoff gap

TR-POLY-09A can persist a deterministic Native V2 root, while B6 expects a validated `NativePolyV2DatasetBuild` object. A real corpus may live outside the repository, so B8A adds a separate fail-closed reload boundary rather than changing the frozen 09A builder.

```text
external/mounted persisted root
        ↓
manifest.json / manifest.sha256 / build.json
        ↓
B8A exact canonical metadata verification
        ↓
TRAIN/VALIDATION target+image set verification
        ↓
V2 roundtrip + PNG + SHA-256 verification
        ↓
reconstructed NativePolyV2DatasetBuild
        ↓
B6
```

B8A requires exact artifact-directory membership. Extra files are rejected. Because only admitted TRAIN/VALIDATION hashes may exist in `targets/` and `images/`, sealed TEST artifact bytes remain absent and are never parsed.

The B8A receipt binds:

- dataset manifest SHA-256;
- deterministic build ID;
- TRAIN sample IDs;
- VALIDATION sample IDs;
- sealed TEST metadata sample IDs;
- persisted target/image artifact counts;
- reload contract version;
- `test_artifact_bytes_accessed = false`;
- `production_authority = false`.

## First real baseline gate

The code path is ready through B7, but the repository does not itself contain a real admitted Native V2 quality corpus. Synthetic regression fixtures are not quality evidence.

Before the first baseline executes, freeze:

1. one physically available Native V2 persisted root;
2. exact manifest SHA-256 and build ID after B8A reload;
3. accepted data rights/provenance/install-pin evidence when external data is involved;
4. exact B4 quality-training recipe;
5. exact B6 execution profile/batch size;
6. explicit B7 descriptor metadata for the complete VALIDATION split;
7. exact B7 decode bound and benchmark identity.

Only then execute:

```text
B6 TRAIN-only candidate creation
        ↓
B5 checkpoint verification
        ↓
B7 full VALIDATION execution
        ↓
B2/B3 failure-strata inspection
        ↓
P09C targeted refinement
```

If the artifact root is unavailable, the baseline remains blocked and no accuracy value is inferred.

## Measurement semantics

Invalid B1 outputs are not filtered. B2 records parse failure plus available sequence/semantic evidence, and B3 keeps invalid/abstain samples in the macro population.

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

No proxy may inherit an unsupported frozen metric ID.

## Evidence hierarchy

```text
SMOKE EVIDENCE
08A/08B: bounded plumbing/trainability

QUALITY TRAINING EVIDENCE
B4/B5/B6: multi-epoch candidate creation and verified checkpoint

QUALITY VALIDATION EVIDENCE
B7 + B1/B2/B3: exact checkpoint on exact hash-bound VALIDATION population

ARTIFACT ADMISSION EVIDENCE
B8A: persisted Native V2 root independently reloaded and verified

SEALED TEST EVIDENCE
Stage 9 only: final one-shot TEST decision
```

None of these automatically grants production authority.

## Required order from here

```text
B8A exact-head CI + merge
        ↓
locate/provide admitted real Native V2 artifact root
        ↓
B8A exact manifest/build preflight
        ↓
freeze B4/B6 recipe + complete B7 descriptors
        ↓
B6 train + B5 persist/reload
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
- Different benchmark/candidate identities cannot be mixed.
- External data requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- No dataset reload, training run, checkpoint or VALIDATION benchmark grants production authority.
- Exact-head green CI is required before merge.
