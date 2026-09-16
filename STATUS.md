# ST-OMR Training Lab Status

Updated: 2026-09-16

This is the current stage-status source. `ARCHITECTURE.md` preserves historical detail; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- latest merged package: PR #163 — TR-POLY-09B8B first real experiment-recipe freeze
- latest verified `main`: `5fb23696d60fc7c3ace7abfbf80991b39cf4e11f`
- B8B exact PR head `e81019573de4094784c23cc520214babbf89c4e8`: CI run #707 success before merge
- active package: TR-POLY-09B8I real-corpus lineage admission bridge
- frozen ≤2-step smoke trainer/checkpoint remain unchanged
- TEST: sealed
- ScoreMosaic / production authority: not granted

## Current stage status

| Stage / package | Description | Status |
|---|---|---|
| 0–6 | Deterministic symbolic → rendered → validated synthetic dataset | Closed / preserved |
| 7-A/B/C | Baseline model/training evidence | Historical baseline |
| 7-D | Specialist architecture/evidence | Historical evidence |
| TR-POLY-02 | Evaluation taxonomy + benchmark identity | Frozen |
| TR-POLY-03/04 | External-data registry + benchmark harness | Closed |
| TR-POLY-05/06 | V2 representation + tokenizer/roundtrip | Frozen |
| TR-POLY-07/08 | Registry + tiny 2D Transformer | Research implementation |
| TR-POLY-08A | ≤2-step deterministic smoke trainer | Frozen / preserved |
| TR-POLY-08B | ≤2-step exact research checkpoint | Frozen / preserved |
| TR-POLY-08C | Stage 6 V1→V2 execution | Single-voice evidence only |
| TR-POLY-09A | Native explicit V2 dataset/materialization | Merged |
| TR-POLY-09B1 | Free-running greedy inference | Merged |
| TR-POLY-09B2 | Deterministic V2 metric/adaptor layer | Merged |
| TR-POLY-09B3 | VALIDATION aggregation by voice/robustness | Merged |
| TR-POLY-09B4 | Multi-epoch TRAIN-only quality-training regime | Merged |
| TR-POLY-09B5 | Separate verified quality-checkpoint artifact + B1 bridge | Merged |
| TR-POLY-09B6 | Full native TRAIN/VALIDATION multi-batch quality execution | Merged |
| TR-POLY-09B7 | Exact descriptor-bound quality VALIDATION B1→B2→B3 execution | Merged / CI green |
| TR-POLY-09B8A | Persisted Native V2 reload + baseline artifact preflight | Merged / CI green |
| TR-POLY-09B8B | Dataset-specific experiment recipe + descriptor/decode/step freeze | Merged / CI green |
| TR-POLY-09B8I | Stage 8 admitted real data ↔ Native V2 lineage admission | Active package |
| First measured quality baseline | Real admitted B6 checkpoint + B7 full VALIDATION run | Artifact-blocked |
| Missing metric admission | Five frozen metrics remain unsupported | Separate work |
| P09C | Evidence-driven refinement | After measured quality evidence |
| P09D | Final candidate/evaluation freeze | Locked |
| Stage 9 | One-shot sealed TEST decision | TEST sealed |
| Stage 10 | ScoreMosaic shadow/integration | Not started |

## Executable quality path

```text
Admitted Stage 8 real TRAIN + VALIDATION metadata/byte receipts
        ↓
Persisted Native V2 TRAIN + VALIDATION root
        ↓
B8A fail-closed reload / identity preflight
        ↓
B8I exact real-data ↔ Native V2 lineage admission
        ↓
complete explicit B7 descriptor metadata
        ↓
B8B exact experiment-recipe freeze
        ↓
B6 deterministic full-population batching
        ↓
B4 multi-epoch TRAIN-only optimization
        ↓
B5 verified selected-state checkpoint
        ↓
B7 exact VALIDATION descriptor/artifact binding
        ↓
B1 free-running inference
        ↓
B2 per-sample metrics
        ↓
B3 overall + voice + robustness aggregation
```

The remaining blocker is not an execution-code gap in B1–B8B. It is the absence of a physically available, admitted real corpus. The connected `ScoreMosaic_Teacher_Gold` Drive hierarchy currently contains empty organizational folders only: no admitted registry records, source corpus, teacher-verified corpus, benchmark output, or Native V2 persisted root is available.

Repository regression fixtures are test evidence only. They must not be reported as real model-quality evidence.

## B8A artifact gate

TR-POLY-09A persists:

```text
manifest.json
manifest.sha256
build.json
targets/<sha256>.json       # TRAIN/VALIDATION only
images/<sha256>.png          # TRAIN/VALIDATION only
```

B8A independently reconstructs and verifies the persisted build while keeping TEST artifact bytes absent/unread.

A real baseline may continue only after preflight yields an exact dataset manifest SHA-256, deterministic build ID, full TRAIN/VALIDATION population, sealed TEST metadata population without TEST bytes, and accepted provenance/license evidence where external data is involved.

## B8I real-corpus admission gate

B8I reuses the frozen Stage 8-0/8-1 real-data safety path rather than inventing a second rights/provenance system. A successful admission requires:

- an admitted `RealDataManifest` with no TEST records;
- exactly one Stage 8-1 byte receipt per admitted real sample;
- exact one-to-one real-data ↔ Native V2 development bindings;
- split, family, image SHA-256 and image-dimension equality;
- one V2 conversion-profile SHA-256 per binding;
- one independent V2 target-review evidence SHA-256 per binding;
- no Stage 8 near-duplicate leakage veto;
- exact B8A TRAIN/VALIDATION population coverage.

A successful B8I receipt grants quality-training eligibility only. It does not grant production authority or commercial-use authority and does not open TEST.

## B8B experiment gate

B8B freezes the first real baseline before quality evidence is observed. The default recipe reuses the merged contracts:

- `FROZEN_POLY_2D_CONFIG`;
- `FROZEN_POLY_2D_QUALITY_CONFIG`;
- 8 epochs;
- AdamW, no scheduler;
- max 8192 optimizer steps;
- validation every epoch;
- minimum mean VALIDATION-loss checkpoint selection, earliest exact tie;
- B6 batch size ≤8;
- full TRAIN and full VALIDATION populations only;
- B7 decode bound at least the longest admitted VALIDATION target and within the model target boundary.

The recipe hash-binds repository SHA, B8A receipt, dataset manifest/build identity, exact split populations/families, batch plan, model/trainer/materialization/B6 profiles, descriptor manifest, B7 split manifest, benchmark identity and decode bound.

Freeze fails if the optimizer-step ceiling cannot cover every full TRAIN batch across every frozen epoch. This prevents silent partial training on a corpus that is larger than the accepted recipe can execute.

## Current metric surface

Numeric B2/B3 metrics:

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

Explicitly unsupported:

```text
musicxml_validity
tedn
notehead_stem_f1
beam_relation_f1
tie_relation_f1
```

No unsupported metric receives a proxy number.

## Required order from here

1. finish B8I code/tests/docs and exact-head CI; merge only if green;
2. populate/provide one rights/provenance-admitted real TRAIN/VALIDATION corpus and Stage 8-1 byte receipts;
3. materialize one Native V2 persisted root and verify it through B8A;
4. emit the exact B8I lineage-admission receipt;
5. complete explicit B7 descriptors and emit the real B8B recipe fingerprint;
6. execute B6 to create the first real quality checkpoint;
7. independently verify the B5 checkpoint round trip;
8. execute B7 over the complete admitted VALIDATION population;
9. inspect B2/B3 failures by voice/robustness and choose P09C from evidence;
10. admit the five missing metric implementations without proxies;
11. freeze P09D candidate/evaluation identity;
12. open sealed TEST once at Stage 9;
13. only then consider Stage 10 ScoreMosaic shadow integration.

## Safety invariants

- TEST remains sealed until Stage 9.
- TRAIN alone changes model parameters.
- VALIDATION is read-only.
- Smoke contracts remain immutable.
- Quality work uses separate versioned provenance.
- No semantic truncation is allowed.
- Invalid/abstain predictions stay visible in denominators.
- Unsupported metrics stay unsupported.
- Benchmark identity and candidate identity must match exactly.
- External data requires rights/license/install-pin admission.
- Teacher corrections and ScoreMosaic uploads are not automatic training data.
- B8I admission is not commercial-use authorization.
- No training or VALIDATION benchmark package grants production authority.
- Every merge requires exact-head green CI.
