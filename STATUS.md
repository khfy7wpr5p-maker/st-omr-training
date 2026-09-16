# ST-OMR Training Lab Status

Updated: 2026-09-16

This is the current stage-status source. `ARCHITECTURE.md` preserves historical detail; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- latest merged package: PR #165 — TR-POLY-09B8J first real baseline start permit
- latest verified `main`: `37647922b098cbe3c46e13e1207eda5f23d8da82`
- B8J exact PR head `288ed731b74767d0ad4aa21ae79a6e506e1a4223`: CI run #711 success before merge
- active package: TR-POLY-09B8K OSSQ-OMR external source selection/admission boundary
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
| TR-POLY-09B8I | Stage 8 admitted real data ↔ Native V2 lineage admission | Merged / CI green |
| TR-POLY-09B8J | Exact B8A+B8I+B8B start permit + authorized full-population B6 wrapper | Merged / CI green |
| TR-POLY-09B8K | OSSQ-OMR source selection + scanned-rights boundary | Active package |
| First measured quality baseline | Real admitted B6 checkpoint + B7 full VALIDATION run | Corpus-rights/artifact blocked |
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
B8J exact baseline start permit
        ↓
B8J authorized full-population B6 wrapper
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

The remaining blocker is not an execution-code gap in B1–B8J. B8K has identified OSSQ-OMR as the primary external source family, but no external corpus is yet admitted or install-pinned. The connected `ScoreMosaic_Teacher_Gold` Drive hierarchy remains empty. OSSQ-OMR scanned-image bytes remain blocked until their per-score IMSLP source/provenance and rights evidence pass the existing Stage 8 admission rules.

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

## B8J baseline-start gate

B8J requires B8A, B8I and B8B to bind the same exact real baseline before B6 can start through the first-baseline path. The permit binds:

- repository SHA;
- dataset manifest/build identity;
- B8A preflight fingerprint;
- B8I admission fingerprint;
- B8B recipe fingerprint;
- exact TRAIN/VALIDATION sample populations;
- deterministic batch plan;
- model/trainer/materialization/B6 execution profiles;
- epoch and optimizer-step budget.

The authorized B8J execution wrapper forces `max_train_samples=None` and `max_validation_samples=None`. Prefix/subsample execution therefore cannot be used for the first measured baseline. Runtime configs are revalidated before B6 and B6 evidence is revalidated after execution.

B8J does not open TEST and grants neither production nor commercial-use authority.

## B8K external-source boundary

B8K pins the reviewed OSSQ-OMR camera-ready source snapshot:

`MALerLab/ossq-omr@7a17e45cddc0b7064fc3a179b62caeb57595e993`

The source family is deliberately split into two registry components:

- annotation sources plus publisher-created synthetic/derived artifacts: CC0, `LICENSE_VERIFIED`, not install-pinned;
- IMSLP-derived scanned-image track: `LICENSE_REVIEW_REQUIRED`, no training/evaluation/commercial permission asserted until per-score upstream rights are independently reviewed.

B8K does not download or admit corpus bytes and does not change TEST, production, or commercial authority.

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

1. finish B8K source-selection code/tests/docs and exact-head CI; merge only if green;
2. perform a deterministic metadata-first rights/provenance audit over the OSSQ-OMR scanned source inventory;
3. acquire only approved scanned artifacts and pin exact SHA-256 identities;
4. create Stage 8-1 byte receipts and admitted TRAIN/VALIDATION metadata;
5. materialize one Native V2 persisted root and verify it through B8A;
6. emit the exact B8I lineage-admission receipt;
7. complete explicit B7 descriptors and emit the real B8B recipe fingerprint;
8. emit the exact B8J baseline start permit;
9. execute the authorized B8J→B6 full-population path to create the first real quality checkpoint;
10. independently verify the B5 checkpoint round trip;
11. execute B7 over the complete admitted VALIDATION population;
12. inspect B2/B3 failures by voice/robustness and choose P09C from evidence;
13. admit the five missing metric implementations without proxies;
14. freeze P09D candidate/evaluation identity;
15. open sealed TEST once at Stage 9;
16. only then consider Stage 10 ScoreMosaic shadow integration.

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
- B8J permit authorizes the exact full first-baseline run only.
- B8K scanned-source selection is not permission to use IMSLP-derived bytes.
- No training or VALIDATION benchmark package grants production authority.
- Every merge requires exact-head green CI.
