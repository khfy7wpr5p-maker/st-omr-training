# ST-OMR Training Lab Status

Updated: 2026-09-16

This is the current stage-status source. `ARCHITECTURE.md` preserves historical detail; `ARCHITECTURE_CURRENT.md` and `ARCHITECTURE_POLYPHONIC_V2_CURRENT.md` describe the active lane.

## Current repository phase

- latest merged package: PR #168 — TR-POLY-09B8M first OSSQ independent rights-evidence batch
- latest verified `main`: `7fe9fd2f0ec17e0ef0141054cb0af573c4c4b912`
- B8M exact PR head `42dc1e2105ae3037e502017866af9ccf1f5b24d8`: CI run #717 success before merge
- active package: TR-POLY-09B8N exact OSSQ source-PDF byte pinning
- first B8N discovery receipt: `9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`
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
| TR-POLY-09B8K | OSSQ-OMR source selection + scanned-rights boundary | Merged / CI green |
| TR-POLY-09B8L | OSSQ per-score source inventory + independent rights/provenance review gate | Merged / CI green |
| TR-POLY-09B8M | First 5 IMSLP sources / 8 OSSQ scores independently reviewed for research training | Merged / CI green |
| TR-POLY-09B8N | Exact PDF source-byte SHA-256 pin + live revalidation | Active package |
| First measured quality baseline | Real admitted B6 checkpoint + B7 full VALIDATION run | Pairing/admission blocked |
| Missing metric admission | Five frozen metrics remain unsupported | Separate work |
| P09C | Evidence-driven refinement | After measured quality evidence |
| P09D | Final candidate/evaluation freeze | Locked |
| Stage 9 | One-shot sealed TEST decision | TEST sealed |
| Stage 10 | ScoreMosaic shadow/integration | Not started |

## Executable quality path

```text
Independently reviewed OSSQ source identity / rights evidence
        ↓
Exact source-PDF SHA-256 byte pin
        ↓
Reviewed image + MusicXML pairing under Stage 8 quarantine
        ↓
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

The remaining blocker is no longer an execution-code gap in B1–B8J, and the first OSSQ source batch now has independent rights evidence plus discovered exact source-PDF identities. The next real-data blocker is construction and independent review of the image/MusicXML pairs that will become Stage 8 TRAIN/VALIDATION candidates. The connected `ScoreMosaic_Teacher_Gold` Drive hierarchy remains empty; no admitted Native V2 real corpus exists yet.

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

## B8L per-score rights/provenance gate

B8L binds the camera-ready OSSQ source inventory to the exact upstream metadata objects:

- `data/scanned_score_types.tsv` Git blob `35ebe84f0c9f03a41231c92fa21183242bf12774`;
- `data/scores_w_pub.yaml` Git blob `7a72b220f4faa897900f51bcb85e49b845cb1304`.

The pinned scanned-score table has 122 score entries across 116 unique work paths. B8L recomputes the Git blob SHA from caller-supplied TSV bytes before parsing them.

Every real-image score type (`0`, `1`, `1\``, `1\`\``, `3`, `4`) requires an exact score-ID/IMSLP-ID/source-type review record. Approved records require separate provenance evidence and separate independent rights evidence plus explicit training/commercial/redistribution permission booleans. The upstream OSSQ copyright label itself cannot serve as the independent rights evidence.

Scores sharing one IMSLP source file may not carry contradictory rights/provenance decisions. Pending or rejected records remain blocked. B8L only identifies later Stage 8 candidates; it does not download source bytes, grant production authority, grant final commercial-use authority, or open TEST.

## B8M first independent rights-evidence batch

B8M independently reviewed five exact IMSLP sources covering eight OSSQ score records:

- `#04047` → `7397765`;
- `#04755` → `8071278`;
- `#64136` → `7103818`;
- `#64141` → `7070781`, `7075297`, `7078259`, `7093885`;
- `#242305` → `7108150`.

For these records only, B8M records `PUBLIC_DOMAIN`, independent rights review and research-training candidacy. It deliberately keeps commercial use and redistribution false. Every other scanned-track OSSQ record remains pending and blocked.

B8M grants neither Stage 8 admission nor production/commercial authority and opens no TEST bytes.

## B8N exact source-byte pin

B8N fetches only the five B8M-reviewed source-PDF URLs and validates actual source bytes before hashing. Non-PDF responses, population drift and sources over the 64 MiB source-document limit fail closed.

The first discovery run emitted canonical receipt:

`9f9b678e2365ec849cc19424b28d8dbdb435af3a5a9ef5b47cf7a460e72a801c`

The hash-only receipt is committed as `evidence/ossq_b8n_source_byte_pins.json`. The dedicated CI workflow then re-fetches the same five exact source URLs and must prove that all SHA-256 values and byte counts still match before B8N can merge.

Raw PDF bytes are neither committed nor uploaded as workflow artifacts. B8N grants no Stage 8 admission, TEST, production or commercial authority.

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

1. finish B8N live receipt verification on the exact PR head and merge only if both standard CI and source-byte CI are green;
2. derive/select image regions and exact MusicXML targets from the byte-pinned OSSQ source batch, with independent pairing review;
3. assign leakage-safe TRAIN/VALIDATION families and create Stage 8 quarantined records;
4. validate exact source/image/MusicXML bytes and emit Stage 8-1 receipts, then admit only reviewed development records;
5. expand the OSSQ rights/byte-pinned batch before recipe freeze if the first batch is insufficient for voice/robustness coverage;
6. materialize one Native V2 persisted root and verify it through B8A;
7. emit the exact B8I lineage-admission receipt;
8. complete explicit B7 descriptors and emit the real B8B recipe fingerprint;
9. emit the exact B8J baseline start permit;
10. execute the authorized B8J→B6 full-population path to create the first real quality checkpoint;
11. independently verify the B5 checkpoint round trip;
12. execute B7 over the complete admitted VALIDATION population;
13. inspect B2/B3 failures by voice/robustness and choose P09C from evidence;
14. admit the five missing metric implementations without proxies;
15. freeze P09D candidate/evaluation identity;
16. open sealed TEST once at Stage 9;
17. only then consider Stage 10 ScoreMosaic shadow integration.

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
- B8L upstream copyright labels are not independent rights approval.
- B8M grants research-training candidacy only to the explicitly reviewed first batch; no commercial/redistribution authority.
- B8N source-byte pins are hash-only evidence, not Stage 8 sample admission.
- No training or VALIDATION benchmark package grants production authority.
- Every merge requires exact-head green CI.
